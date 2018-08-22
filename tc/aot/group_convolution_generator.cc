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

DEFINE_uint32(
    number_options,
    10,
    "Number of options per input set to generate (default: 10");
DEFINE_uint32(
    number_inputs,
    100,
    "Number of different input sets to generate (default: 100");
DEFINE_string(
    output,
    "kernels.proto",
    "Output filename (default: kernels.proto");

DEFINE_uint32(B0, 1, "B0");
DEFINE_uint32(B1, 1, "B1");
DEFINE_uint32(B2, 1, "B2");
DEFINE_uint32(G0, 1, "G0");
DEFINE_uint32(G1, 1, "G1");
DEFINE_uint32(G2, 1, "G2");
DEFINE_uint32(T0, 1, "T0");
DEFINE_uint32(T1, 1, "T1");
DEFINE_uint32(T2, 1, "T2");
DEFINE_uint32(T3, 1, "T5");
DEFINE_uint32(T4, 1, "T4");
DEFINE_uint32(T5, 1, "T5");

DEFINE_uint32(threads, 1, "Number of threads.");

} // namespace

uint64_t threadsPerBlock(const tc::Block& b) {
  return b.view.proto.x() * b.view.proto.y() * b.view.proto.z();
}
uint64_t blocksPerGrid(const tc::Grid& g) {
  return g.view.proto.x() * g.view.proto.y() * g.view.proto.z();
}

bool stillGoodAfterTighening(const tc::CudaCompilationResult& res) {
  auto t = threadsPerBlock(res.block);
  if (t < 32)
    return false;
  auto b = blocksPerGrid(res.grid);

  if (b < 20)
    return false;
  return true;
}

auto loadProto(const std::string& filename) {
  tc::AotBuf kis;
  std::ifstream in{filename, std::ios::binary};
  if (not kis.ParseFromIstream(&in)) {
    throw std::invalid_argument{"Could input parse protobuf."};
  }
  return kis;
}

int main(int argc, char* argv[]) {
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  tc::AotBuf kis;
  if (fs::exists(FLAGS_output)) {
    std::cout << FLAGS_output << " already exists. Will reload and override."
              << std::endl;
    kis = loadProto(FLAGS_output);
  }

  auto gc_tc = tc::makeGroupConvolution2DTc(1, 1);

  std::unordered_set<uint64_t> used_ids;
  for (const auto& ki : kis.kernels()) {
    if (ki.id() != 0) {
      used_ids.insert(ki.id());
    }
  }

  uint64_t id = 0;

  std::atomic<uint64_t> tries{0};
  std::atomic<uint64_t> successes{0};
  using namespace std;
  using namespace chrono;
  std::mutex mtx_ids;
  std::vector<std::thread> workers;
  uint64_t total = FLAGS_number_options * FLAGS_number_inputs;

  tc::OptionsAndInputsGenerator<tc::GCInputsGenerator> gen{
      FLAGS_number_inputs, FLAGS_number_options, 3, 2};

  std::vector<std::vector<tc::TensorInfo>> inputs;
  std::generate_n(
      std::back_inserter(inputs), FLAGS_number_inputs, tc::GCInputsGenerator{});
  std::vector<std::vector<tc::KernelInfo>> generated_kernels{
      FLAGS_number_inputs};
  std::vector<std::mutex> data_mtxs{FLAGS_number_inputs};

  for (int64_t t = 0; t < FLAGS_threads; ++t) {
    workers.emplace_back([&gc_tc,
                          &tries,
                          &successes,
                          total,
                          &id,
                          &used_ids,
                          &mtx_ids,
                          &inputs,
                          &generated_kernels,
                          &data_mtxs]() {
      while (true) {
        auto idx = FLAGS_number_inputs;
        for (uint64_t i = 0; i < FLAGS_number_inputs; ++i) {
          std::lock_guard<std::mutex> lock{data_mtxs[i]};
          if (generated_kernels[i].size() < FLAGS_number_options) {
            idx = i;
            break;
          }
        }
        if (idx == FLAGS_number_inputs) {
          break;
        }

        auto DLU = tc::makeDLConstTensorVector(inputs[idx]);
        auto DL = tc::extractRawPtrs(DLU);
        auto outputsInfo =
            tc::inferOutputTensorInfo(gc_tc, "group_convolution", DL);

        tc::OptionsGenerator og{inputs[idx], 3, 2};

        while (true) {
          {
            std::lock_guard<std::mutex> lock{data_mtxs[idx]};
            if (generated_kernels[idx].size() >= FLAGS_number_options) {
              break;
            }
          }

          std::cout << "Compilation attempts: " << tries.fetch_add(1)
                    << " Successes: " << successes.load() << std::endl;

          auto options = og();
          tc::CudaBackend::CompilationResultType res;
          auto t0 = high_resolution_clock::now();
          try {
            res = tc::compileToSource<tc::CudaBackend>(
                gc_tc, "group_convolution", DL, options, true);
          } catch (std::exception& e) {
            std::cout << "Something went wrong: " << e.what() << std::endl;
            continue;
          }
          auto t1 = high_resolution_clock::now();
          auto compilation_time = t1 - t0;
          std::cout << "Compilation time: "
                    << duration_cast<milliseconds>(compilation_time).count()
                    << "ms" << std::endl;
          if (not stillGoodAfterTighening(res)) {
            continue;
          }
          ++successes;
          std::lock_guard<std::mutex> lock_ids{mtx_ids};
          while (used_ids.count(id) > 0) {
            ++id;
          }
          used_ids.insert(id);
          auto ki = makeKernelInfo(
              res,
              id,
              gc_tc,
              inputs[idx],
              outputsInfo,
              options,
              compilation_time);

          std::lock_guard<std::mutex> lock_data{data_mtxs[idx]};
          generated_kernels[idx].push_back(std::move(ki));
        }
      }
    });
  }

  for (auto& t : workers) {
    t.join();
  }

  for (const auto& kernel : generated_kernels) {
    for (const auto& ki : kernel) {
      *kis.add_kernels() = ki;
    }
  }
  std::ofstream output{FLAGS_output, std::ios::binary | std::ios::trunc};
  if (not kis.SerializeToOstream(&output)) {
    std::cout << "Serialization failed" << std::endl;
  }

  return 0;
}
