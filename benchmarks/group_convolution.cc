#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "tc/aten/aten.h"
#include "tc/aten/aten_autotuner.h"
#include "tc/core/cuda/cuda.h"
#include "tc/core/cuda/cuda_backend.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/cuda/cuda_tc_executor.h"
#include "tc/core/flags.h"

DEFINE_uint32(N, 32, "Batch size (NCHW notation)");
DEFINE_uint32(G, 32, "Number of groups (NCHW notation)");
DEFINE_uint32(C, 4, "Input channels (NCHW notation)");
DEFINE_uint32(F, 4, "Output filters (NCHW notation)");
DEFINE_uint32(H, 56, "Image width (NCHW notation)");
DEFINE_uint32(W, 56, "Image height (NCHW notation)");
DEFINE_uint32(KH, 3, "Kernel width (NCHW notation)");
DEFINE_uint32(KW, 3, "Kernel height (NCHW notation)");
DEFINE_string(save_tuner_proto_prefix, ".", "Save protobuf prefix.");

int main(int argc, char** argv) {
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  tc::aten::setAtenSeed(tc::initRandomSeed(), at::Backend::CUDA);

  std::string tc = R"(
def group_convolution(float(N,G,C,H,W) I, float(G,F,C,KH,KW) W1, float(G,F) B)
-> (O)
{
    O(n, g, f, h, w) +=!
        I(n, g, r_c, h + r_kh, w + r_kw) * W1(g, f, r_c, r_kh, r_kw)
    O(n, g, f, h, w)  = O(n, g, f, h, w) + B(g, f)
}
)";

  std::string suffix = std::string("_N_") + std::to_string(FLAGS_N) +
      std::string("_G_") + std::to_string(FLAGS_G) + std::string("_C_") +
      std::to_string(FLAGS_C) + std::string("_F_") + std::to_string(FLAGS_F) +
      std::string("_W_") + std::to_string(FLAGS_W) + std::string("_H_") +
      std::to_string(FLAGS_H) + std::string("_KW_") + std::to_string(FLAGS_KW) +
      std::string("_KH_") + std::to_string(FLAGS_KH);

  std::vector<at::Tensor> inputs{
      at::CUDA(at::kFloat).rand({FLAGS_N, FLAGS_G, FLAGS_C, FLAGS_H, FLAGS_W}),
      at::CUDA(at::kFloat)
          .rand({FLAGS_G, FLAGS_F, FLAGS_C, FLAGS_KH, FLAGS_KW}),
      at::CUDA(at::kFloat).rand({FLAGS_G, FLAGS_F})};

  // If num threads is too small just get some better default
  auto threads = (FLAGS_W >= 10) ? std::vector<size_t>{FLAGS_W / 4, FLAGS_H / 2}
                                 : std::vector<size_t>{4, 8, 4};
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .tile(1, 1, 7, 7)
                     .mapToThreads(threads)
                     .mapToBlocks({32, 32})
                     .useSharedMemory(false)
                     .usePrivateMemory(false)
                     .unroll(2);

  tc::aten::ATenAutotuner<tc::CudaBackend, tc::autotune::GeneticSearch>
      geneticAutotuneATen(tc);

  tc::autotune::TuningParameterFixer fixer;
  fixer.fixOuterScheduleFusionStrategy(tc::FusionStrategy::Max)
      .fixIntraTileScheduleFusionStrategy(tc::FusionStrategy::Max)
      .fixFixParametersBeforeScheduling(true)
      .fixUnrollFactor(2)
      .fixTilingParameters({1, 1, 7, 7})
      .fixTileImperfectlyNested(false)
      .fixUseSharedMemory(false)
      .fixUsePrivateMemory(false)
      .fixUnrollCopyShared(false)
      .fixMatchLibraryCalls(false);

  geneticAutotuneATen.tune(
      "group_convolution",
      inputs,
      options,
      FLAGS_save_tuner_proto_prefix + std::string("/group_convolution_cache") +
          suffix,
      fixer);
}
