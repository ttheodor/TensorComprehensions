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
#include "tc/proto/aot.pb.h"
#include "tc/version/version.h"

namespace {
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

constexpr static auto TC_MLP3_NAME = "mlp3";
constexpr static auto TC_MLP3 = R"TC(
def mlp3(float(B,N) I, float(O,N) W2, float(O) B2, float(P,O) W3, float(P) B3,
         float(Q,P) W4, float(Q) B4) -> (O2, O3, O4) {
    O2(b, o) +=!  I(b, n) * W2(o, n)
    O2(b, o)  =  O2(b, o) + B2(o)
    O2(b, o)  = fmax(O2(b, o), 0)
    O3(b, p) +=! O2(b, o) * W3(p, o)
    O3(b, p)  =  O3(b, p) + B3(p)
    O3(b, p)  = fmax(O3(b, p), 0)
    O4(b, q) +=! O3(b, p) * W4(q, p)
    O4(b, q)  =  O4(b, q) + B4(q)
    O4(b, q)  = fmax(O4(b, q), 0)
}
)TC";

int main(int argc, char* argv[]) {
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  static tc::AotBuf kis;
  if (fs::exists(FLAGS_output)) {
    std::cout << FLAGS_output << " already exists. Will reload and override."
              << std::endl;
    kis = loadProto(FLAGS_output);
  }

  std::unordered_set<uint64_t> used_ids;
  for (const auto& ki : kis.kernels()) {
    if (ki.id() != 0) {
      used_ids.insert(ki.id());
    }
  }

  uint64_t id = 0;

  std::atomic_size_t tries{0};
  std::atomic_size_t successes{0};
  using namespace std;
  using namespace chrono;
  std::mutex mtx;
  std::vector<std::thread> workers;
  uint64_t total = FLAGS_number_options * FLAGS_number_inputs;

  static auto write_proto = []() {
    std::ofstream output{FLAGS_output, std::ios::binary | std::ios::trunc};
    if (not kis.SerializeToOstream(&output)) {
      std::cout << "Serialization failed" << std::endl;
    }
  };

  tc::OptionsAndInputsGenerator<tc::MLP3InputsGenerator> gen{
      FLAGS_number_inputs, FLAGS_number_options, 3, 1};
  for (int64_t t = 0; t < FLAGS_threads; ++t) {
    workers.emplace_back(
        [&gen, &tries, &successes, total, &id, &used_ids, &mtx]() {
          while (successes.load() < total) {
            std::cout << "Compilation attempts: " << tries.fetch_add(1)
                      << " Successes: " << successes.load() << std::endl;
            try {
              auto [inputs, options] = gen.generate();
              auto DLU = tc::makeDLConstTensorVector(inputs);
              auto DL = tc::extractRawPtrs(DLU);
              auto outputsInfo =
                  tc::inferOutputTensorInfo(TC_MLP3, TC_MLP3_NAME, DL);

              auto t0 = high_resolution_clock::now();
              auto res = tc::compileToSource<tc::CudaBackend>(
                  TC_MLP3, TC_MLP3_NAME, DL, options);
              auto t1 = high_resolution_clock::now();
              auto compilation_time = t1 - t0;
              std::cout << "Compilation time: "
                        << duration_cast<milliseconds>(compilation_time).count()
                        << "ms" << std::endl;
              if (not stillGoodAfterTighening(res)) {
                std::cout << "Not enough threads and/or blocks. Discarding..."
                          << std::endl;
                std::cout << tc::CudaMappingOptionsAsCpp(options) << std::endl;
                std::cout << res.grid.view.proto.x() << ' '
                          << res.grid.view.proto.y() << ' '
                          << res.grid.view.proto.z() << std::endl;
                std::cout << res.block.view.proto.x() << ' '
                          << res.block.view.proto.y() << ' '
                          << res.block.view.proto.z() << std::endl;
                std::cout << res.source << std::endl;
                gen.remove(inputs, options);
                continue;
              }
              ++successes;
              std::lock_guard<std::mutex> lock{mtx};
              while (used_ids.count(id) > 0) {
                ++id;
              }
              used_ids.insert(id);
              *kis.add_kernels() = makeKernelInfo(
                  res,
                  id,
                  TC_MLP3,
                  inputs,
                  outputsInfo,
                  options,
                  compilation_time);
              ++id;
              if (successes.load() % 100 == 0) {
                write_proto();
              }
            } catch (...) {
              break;
            }
          }
        });
  }

  auto handler = [](int) {
    write_proto();
    std::abort();
  };

  std::signal(SIGINT, handler);
  std::signal(SIGTERM, handler);
  std::signal(SIGKILL, handler);

  for (auto& t : workers) {
    t.join();
  }
  write_proto();

  return 0;
}
