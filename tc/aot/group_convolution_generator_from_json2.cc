#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_set>

#if __cpp_lib_filesystem >= 201603
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include <gflags/gflags.h>
#include <json.hpp>
#include <pcg_random.hpp>

#include "tc/aot/common.h"
#include "tc/core/compiler.h"
#include "tc/core/cuda/cuda_backend.h"
#include "tc/core/tensor.h"
#include "tc/library/group_convolution.h"
#include "tc/proto/aot.pb.h"
#include "tc/version/version.h"

namespace {
DEFINE_uint32(N, 32, "Batch size (NCHW notation)");
DEFINE_uint32(G, 32, "Number of groups (NCHW notation)");
DEFINE_uint32(C, 4, "Input channels (NCHW notation)");
DEFINE_uint32(F, 4, "Output filters (NCHW notation)");
DEFINE_uint32(H, 56, "Image width (NCHW notation)");
DEFINE_uint32(W, 56, "Image height (NCHW notation)");
DEFINE_uint32(KH, 3, "Kernel width (NCHW notation)");
DEFINE_uint32(KW, 3, "Kernel height (NCHW notation)");

DEFINE_string(
    output,
    "kernels.proto",
    "Output filename (default: kernels.proto)");
DEFINE_string(input, "", "Input filename");
DEFINE_uint32(threads, 1, "Number of threads.");
} // namespace

using nlohmann::json;

json read_json(const std::string& input) {
  json j;
  std::ifstream in{input};
  in >> j;
  return j;
}

std::vector<tc::TensorInfo> make_inputs(const json& j) {
  std::vector<int64_t> I_sizes{j.at("size0"),
                               j.at("size1"),
                               j.at("size2"),
                               j.at("size3"),
                               j.at("size4")};
  std::vector<int64_t> W1_sizes{j.at("size5"),
                                j.at("size6"),
                                j.at("size7"),
                                j.at("size8"),
                                j.at("size9")};
  std::vector<int64_t> B_sizes{j.at("size10"), j.at("size11")};
  DLDataType floatType{DLDataTypeCode::kDLFloat, 32, 1};

  return {
      tc::TensorInfo{floatType, 32, I_sizes, tc::makeStridesFromSizes(I_sizes)},
      tc::TensorInfo{
          floatType, 32, W1_sizes, tc::makeStridesFromSizes(W1_sizes)},
      tc::TensorInfo{
          floatType, 32, B_sizes, tc::makeStridesFromSizes(B_sizes)}};
}

uint64_t getInt(const json& j, const std::string& s) {
  return j.at(s);
}

tc::FusionStrategy toStrategy(const std::string& s) {
  if (s == "Min")
    return tc::FusionStrategy::Min;
  if (s == "Max")
    return tc::FusionStrategy::Max;
  if (s == "Preserve3Coincident")
    return tc::FusionStrategy::Preserve3Coincident;
  throw std::invalid_argument{"Unknown strategy"};
}

uint64_t oneIfzero(uint64_t x) {
  return x == 0 ? 1 : x;
}

std::vector<uint64_t> toCudaThreads(const json& i) {
  return {getInt(i, "bx"), getInt(i, "by"), getInt(i, "bz")};
}

std::vector<uint64_t> toCudaBlocks(const json& i) {
  return {getInt(i, "gx"), getInt(i, "gy"), getInt(i, "gz")};
}

tc::CudaMappingOptions make_options(const json& j) {
  return tc::CudaMappingOptions::makeNaiveMappingOptions()
      .outerScheduleFusionStrategy(toStrategy(j.at("outer fusion")))
      .outerScheduleAllowSkewing(true)
      .intraTileScheduleFusionStrategy(toStrategy(j.at("intra tile fusion")))
      .intraTileScheduleAllowSkewing(true)
      .tile(std::vector<uint64_t>{1ul, 1ul, getInt(j, "t2")})
      .mapToThreads(toCudaThreads(j))
      .mapToBlocks(toCudaBlocks(j))
      .tileImperfectlyNested(getInt(j, "tile imperfect"))
      .unroll(getInt(j, "unroll factor"))
      .useSharedMemory(getInt(j, "use shared memory"))
      .unrollCopyShared(getInt(j, "unroll copy shared"))
      .useReadOnlyCache(getInt(j, "user readonly cache"))
      .matchLibraryCalls(false);
}

int main(int argc, char* argv[]) {
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  if (fs::exists(FLAGS_output)) {
    std::cout << FLAGS_output << " already exists." << std::endl;
    exit(1);
  }
  if (not fs::exists(FLAGS_input)) {
    std::cout << "Input file " << FLAGS_input << " does not exist."
              << std::endl;
    exit(1);
  }

  auto gc_tc = tc::makeGroupConvolution2DTc(1, 1);

  auto data = read_json(FLAGS_input);

  std::unordered_set<uint64_t> used_ids;
  uint64_t id = 0;

  std::mutex mtx;
  std::vector<std::thread> workers;

  tc::AotBuf kis;
  std::atomic_uint64_t counter{0};

  for (int64_t t = 0; t < FLAGS_threads; ++t) {
    workers.emplace_back(
        [&gc_tc, &kis, &data, &id, &used_ids, &mtx, &counter]() {
          while (true) {
            if (counter.load() >= data.size())
              break;
            auto idx = counter.fetch_add(1ul);

            if (idx >= data.size()) {
              break;
            }

            try {
              auto inputs = make_inputs(data[idx]);
              auto options = make_options(data[idx]);

              auto DLU = tc::makeDLConstTensorVector(inputs);
              auto DL = tc::extractRawPtrs(DLU);
              auto outputsInfo =
                  tc::inferOutputTensorInfo(gc_tc, "group_convolution", DL);

              using namespace std;
              using namespace chrono;
              auto t0 = high_resolution_clock::now();
              auto res = tc::compileToSource<tc::CudaBackend>(
                  gc_tc, "group_convolution", DL, options);
              auto t1 = high_resolution_clock::now();
              auto compilation_time = t1 - t0;
              std::cout << "Compilation time: "
                        << duration_cast<milliseconds>(compilation_time).count()
                        << "ms" << std::endl;
              std::lock_guard<std::mutex> lock{mtx};
              while (used_ids.count(id) > 0) {
                ++id;
              }
              used_ids.insert(id);

              *kis.add_kernels() = makeKernelInfo(
                  res,
                  id,
                  gc_tc,
                  inputs,
                  outputsInfo,
                  options,
                  compilation_time);
              std::cout << "Compiled " << counter << '/' << data.size()
                        << std::endl;
            } catch (const std::exception& e) {
              std::cout << e.what() << std::endl;
              throw;
            }
          }
        });
  }

  for (auto& t : workers) {
    t.join();
  }

  std::ofstream output{FLAGS_output, std::ios::binary | std::ios::trunc};
  if (not kis.SerializeToOstream(&output)) {
    std::cout << "Serialization failed" << std::endl;
  }

  return 0;
}
