#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_set>

#include <gflags/gflags.h>
#include <pcg_random.hpp>

#include "tc/core/compiler.h"
#include "tc/core/cuda/cuda_backend.h"
#include "tc/core/cuda/cuda_mapping_options.h"
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

DEFINE_uint32(number, 100, "Number samples to generate (default: 100");
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

uint64_t getMaxSize(const std::vector<tc::TensorInfo>& ti) {
  std::vector<int64_t> ms;
  for (const auto& t : ti) {
    ms.push_back(*std::max_element(t.shape.begin(), t.shape.end()));
  }
  return *std::max_element(ms.begin(), ms.end());
}

} // namespace

class OptionsGenerator {
 public:
  OptionsGenerator(const std::vector<tc::TensorInfo>& ti)
      : maxSize{getMaxSize(ti)},
        rng{pcg_extras::seed_seq_from<std::random_device>{}} {}

  tc::CudaMappingOptions operator()() {
    auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                       .outerScheduleFusionStrategy(makeFusionStrategy())
                       .outerScheduleAllowSkewing(true)
                       .intraTileScheduleFusionStrategy(makeFusionStrategy())
                       .intraTileScheduleAllowSkewing(true)
                       .tile(makeTiles())
                       .mapToThreads(makeBlock())
                       .mapToBlocks(makeGrid())
                       .tileImperfectlyNested(makeBool())
                       .unroll(makeUnroll())
                       .useSharedMemory(makeBool());
    options.unrollCopyShared(
        options.proto().use_shared_memory() ? makeBool() : false);

    return options;
  };

 private:
  tc::FusionStrategy makeFusionStrategy() {
    switch (std::uniform_int_distribution<int>{1, 3}(rng)) {
      case 1:
        return tc::FusionStrategy::Max;
      case 2:
        return tc::FusionStrategy::Preserve3Coincident;
      case 3:
        return tc::FusionStrategy::Min;
    }
    return tc::FusionStrategy::Max;
  }

  std::vector<uint64_t> makeTiles() {
    std::vector<uint64_t> sizes(3);
    sizes[0] = 1;
    sizes[1] = 1;
    sizes[2] = std::uniform_int_distribution<uint64_t>{0, maxSize}(rng);

    return sizes;
  }

  uint64_t oneToMaxSize() {
    return std::uniform_int_distribution<uint64_t>{1, maxSize}(rng);
  }
  std::vector<uint64_t> makeCudaDim() {
    std::vector<uint64_t> sizes(3);
    std::generate_n(sizes.begin(), 3, [this]() { return oneToMaxSize(); });
    return sizes;
  }

  std::vector<uint64_t> makeBlock() {
    auto valid = [](const std::vector<uint64_t>& v) {
      return v[0] <= 1024 and v[1] <= 1024 and v[2] <= 64 and
          v[0] * v[1] * v[2] <= 1024;
    };
    auto min32 = [](const std::vector<uint64_t>& v) {
      return v[0] * v[1] * v[2] >= 32;
    };
    while (true) {
      auto v = makeCudaDim();
      if (valid(v) and min32(v))
        return v;
    }
  }

  std::vector<uint64_t> makeGrid() {
    auto check = [](const std::vector<uint64_t>& v) {
      return v[0] < 2147483648 and v[1] < 65536 and v[2] < 65536;
    };
    // there are 56 SMs on a P100
    auto min56 = [](const std::vector<uint64_t>& v) {
      return v[0] * v[1] * v[2] >= 56;
    };
    while (true) {
      auto v = makeCudaDim();
      if (check(v) and min56(v))
        return v;
    }
  }

  bool makeBool() {
    return std::uniform_int_distribution<int>{0, 1}(rng);
  }

  uint64_t makeUnroll() {
    return oneToMaxSize();
  }

  uint64_t maxSize;
  pcg64 rng;
};

struct OptionsHash {
  size_t operator()(const tc::CudaMappingOptions& o) const {
    return std::hash<std::string>{}(o.proto().SerializeAsString());
  }
};

std::vector<tc::CudaMappingOptions> generaterUniqueOptions(
    size_t n,
    const std::vector<tc::TensorInfo>& ti) {
  std::unordered_set<tc::CudaMappingOptions, OptionsHash> options;
  std::generate_n(
      std::inserter(options, options.end()), n, OptionsGenerator{ti});
  return {options.begin(), options.end()};
}

std::vector<tc::TensorInfo> makeTensorInfo() {
  auto N = FLAGS_N;
  auto G = FLAGS_G;
  auto C = FLAGS_C;
  auto F = FLAGS_F;
  auto H = FLAGS_H;
  auto W = FLAGS_W;
  auto KH = FLAGS_KH;
  auto KW = FLAGS_KW;

  std::vector<int64_t> I_sizes{N, G, C, H, W};
  std::vector<int64_t> W1_sizes{G, F, C, KH, KW};
  std::vector<int64_t> B_sizes{G, F};
  DLDataType floatType{DLDataTypeCode::kDLFloat, 32, 1};

  return {
      tc::TensorInfo{floatType, 32, I_sizes, tc::makeStridesFromSizes(I_sizes)},
      tc::TensorInfo{
          floatType, 32, W1_sizes, tc::makeStridesFromSizes(W1_sizes)},
      tc::TensorInfo{
          floatType, 32, B_sizes, tc::makeStridesFromSizes(B_sizes)}};
}

tc::KernelInfo makeKernelInfo(
    const tc::CudaCompilationResult& res,
    const std::string& tc,
    const std::vector<tc::TensorInfo>& inputsInfo,
    const std::vector<tc::TensorInfo>& outputsInfo,
    const tc::CudaMappingOptions& opts,
    const std::chrono::high_resolution_clock::duration& compilation_time) {
  tc::KernelInfo ki;
  ki.set_tc(tc);
  for (const auto& i : inputsInfo) {
    *ki.add_inputs() = i.toProtobuf();
  }
  for (const auto& o : outputsInfo) {
    *ki.add_outputs() = o.toProtobuf();
  }

  *ki.mutable_kernel_options() = opts.proto();
  ki.set_cuda_source(res.source);
  ki.set_specialized_name(res.specializedName);
  *ki.mutable_parameters() = {res.parameters.begin(), res.parameters.end()};
  *ki.mutable_tight_block() = res.block.view.proto;
  *ki.mutable_tight_grid() = res.grid.view.proto;
  ki.set_git_version(tc::git_version);
  ki.set_compilation_time(
      std::chrono::duration_cast<std::chrono::milliseconds>(compilation_time)
          .count());
  ki.set_id(0);
  return ki;
}

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

  if (b < 56)
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
  if (std::filesystem::exists(FLAGS_output)) {
    std::cout << FLAGS_output << " already exists. Will reload and override."
              << std::endl;
    kis = loadProto(FLAGS_output);
  }

  auto gc_tc = tc::makeGroupConvolution2DTc(1, 1);
  auto inputsInfo = makeTensorInfo();
  auto DLU = tc::makeDLConstTensorVector(inputsInfo);
  auto DL = tc::extractRawPtrs(DLU);
  auto outputsInfo = tc::inferOutputTensorInfo(gc_tc, "group_convolution", DL);

  std::atomic_size_t tries{0};
  std::atomic_size_t successes{0};
  using namespace std;
  using namespace chrono;
  std::mutex mtx;
  std::vector<std::thread> workers;
  auto perThreadWork = static_cast<uint64_t>(
      std::llround(FLAGS_number / static_cast<float>(FLAGS_threads)));
  for (int64_t t = 0; t < FLAGS_threads; ++t) {
    auto numberOptions = t < FLAGS_threads - 1
        ? perThreadWork
        : FLAGS_number - t * perThreadWork;
    workers.emplace_back([numberOptions,
                          &inputsInfo,
                          &outputsInfo,
                          &tries,
                          &successes,
                          &gc_tc,
                          &DL,
                          &kis,
                          &mtx]() {
      std::vector<tc::KernelInfo> kernelIs;
      std::unordered_set<tc::CudaMappingOptions, OptionsHash> used_options;
      while (kernelIs.size() < numberOptions) {
        auto options =
            generaterUniqueOptions(numberOptions - kernelIs.size(), inputsInfo);
        for (const auto opts : options) {
          if (used_options.count(opts) > 0) {
            continue;
          }
          std::cout << "Compilation attempts: " << tries.fetch_add(1)
                    << " Successes: " << successes.load() << std::endl;
          auto t0 = high_resolution_clock::now();
          auto res = tc::compile<tc::CudaBackend>(
              gc_tc, "group_convolution", DL, opts);
          auto t1 = high_resolution_clock::now();
          auto compilation_time = t1 - t0;
          std::cout << "Compilation time: "
                    << duration_cast<milliseconds>(compilation_time).count()
                    << "ms" << std::endl;
          if (not stillGoodAfterTighening(res)) {
            // std::cout << "Not enough threads and/or blocks. Discarding... "
            //<< std::endl;
            // std::cout << tc::CudaMappingOptionsAsCpp(opts) << std::endl;
            // std::cout << res.grid.view.proto.x() << ' '
            //<< res.grid.view.proto.y() << ' '
            //<< res.grid.view.proto.z() << std::endl;
            // std::cout << res.block.view.proto.x() << ' '
            //<< res.block.view.proto.y() << ' '
            //<< res.block.view.proto.z() << std::endl;
            // std::cout << res.source << std::endl;
            continue;
          }
          ++successes;

          used_options.insert(opts);
          kernelIs.push_back(makeKernelInfo(
              res, gc_tc, inputsInfo, outputsInfo, opts, compilation_time));
        }
      }
      std::lock_guard<std::mutex> lock{mtx};
      for (auto& k : kernelIs) {
        *kis.add_kernels() = std::move(k);
      }
    });
  }

  for (auto& t : workers) {
    t.join();
  }
  std::unordered_set<uint64_t> used_ids;
  for (const auto& ki : kis.kernels()) {
    if (ki.id() != 0) {
      used_ids.insert(ki.id());
    }
  }
  uint64_t id = 0;
  for (int i = 0; i < kis.kernels_size(); ++i) {
    while (used_ids.count(id) > 0) {
      ++id;
    }
    kis.mutable_kernels(i)->set_id(id++);
  }

  std::ofstream output{FLAGS_output, std::ios::binary | std::ios::trunc};
  if (not kis.SerializeToOstream(&output)) {
    std::cout << "Serialization failed" << std::endl;
    return 2;
  }
  return 0;
}
