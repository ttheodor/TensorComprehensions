#include <optional>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "tc/aten/aten_compiler.h"
#include "tc/core/compiler.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/cuda/cuda_tc_executor.h"

using namespace tc;

class OptionsGenerator {
 public:
  CudaMappingOptions operator()();
};

class BenchmarkResult {};

class CompilationQueue {
 public:
  CompilationQueue(size_t n) {
    for (size_t i = 0; i < n; ++i)
      workers.emplace_back([this]() { workerLoop(); });
  }

  ~CompilationQueue() {
    stop.store(true);
    for (auto& worker : workers)
      worker.join();
  }

  using JobTy = std::function<std::unique_ptr<CudaTcExecutor>()>;
  void addJob(JobTy f) {
    std::lock_guard<std::mutex> lock{jobs_mtx};
    jobs.push_back(std::move(f));
  }

  std::optional<std::unique_ptr<CudaTcExecutor>> nextResult() {
    std::unique_lock<std::mutex> lock{results_mtx, std::defer_lock_t{}};
    while (true) {
      lock.lock();
      if (results.empty()) {
        lock.unlock();
        if (totalJobs == totalReturnedResults)
          return {};
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
        continue;
      }
      auto res = std::move(results.front());
      results.pop_front();
      ++totalReturnedResults;
      return res;
    }
  }

 private:
  void workerLoop();

  std::vector<std::thread> workers;
  std::atomic_bool stop{false};

  std::mutex jobs_mtx;
  std::list<JobTy> jobs;

  std::mutex results_mtx;
  std::list<std::unique_ptr<CudaTcExecutor>> results;

  std::atomic_size_t totalJobs{0};
  std::atomic_size_t totalReturnedResults{0};
};

class Sampler {
 public:
  Sampler(
      OptionsGenerator&& g,
      std::string tc,
      std::string entryPoint,
      std::vector<at::Tensor>&& inputs);

  std::vector<BenchmarkResult> benchmarkN(size_t n) {
    generateN(n);
    std::vector<BenchmarkResult> res;
    while (auto exec = nextExec()) {
      if (not*exec)
        continue;
      res.push_back(benchmark(**exec));
    }

    return res;
  }

 private:
  BenchmarkResult benchmark(CudaTcExecutor& exec);

  std::optional<std::unique_ptr<CudaTcExecutor>> nextExec();

  void compile(CudaMappingOptions&& options) {
    q.addJob([this, options = std::move(options)]() {
      return aten::compile<CudaBackend>(tc, entryPoint, inputs, options);
    });
  }

  void generateN(size_t n) {
    for (size_t i = 0; i < n; ++i)
      this->compile(g());
  }

  OptionsGenerator g;
  std::string tc;
  std::string entryPoint;
  std::vector<at::Tensor> inputs;
  std::vector<at::Tensor> outputs;

  CompilationQueue q;
};

namespace {
DEFINE_uint32(N, 32, "Batch size (NCHW notation)");
DEFINE_uint32(G, 32, "Number of groups (NCHW notation)");
DEFINE_uint32(C, 4, "Input channels (NCHW notation)");
DEFINE_uint32(F, 4, "Output filters (NCHW notation)");
DEFINE_uint32(H, 56, "Image width (NCHW notation)");
DEFINE_uint32(W, 56, "Image height (NCHW notation)");
DEFINE_uint32(KH, 3, "Kernel width (NCHW notation)");
DEFINE_uint32(KW, 3, "Kernel height (NCHW notation)");
} // namespace

int main(int argc, char* argv[]) {
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);

  std::string tc = R"(
def group_convolution(float(N,G,C,H,W) I, float(G,F,C,KH,KW) W1, float(G,F) B)
-> (O)
{
    O(n, g, f, h, w) +=!
        I(n, g, r_c, h + r_kh, w + r_kw) * W1(g, f, r_c, r_kh, r_kw)
    O(n, g, f, h, w)  = O(n, g, f, h, w) + B(g, f)
}
)";

  Sampler s{OptionsGenerator{},
            tc,
            "group_convolution",
            std::vector<at::Tensor>{
                at::CUDA(at::kFloat)
                    .rand({FLAGS_N, FLAGS_G, FLAGS_C, FLAGS_H, FLAGS_W}),
                at::CUDA(at::kFloat)
                    .rand({FLAGS_G, FLAGS_F, FLAGS_C, FLAGS_KH, FLAGS_KW}),
                at::CUDA(at::kFloat).rand({FLAGS_G, FLAGS_F})}};

  s.benchmarkN(100);
}
