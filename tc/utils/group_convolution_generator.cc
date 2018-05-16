#include <gflags/gflags.h>

#include "tc/core/compiler.h"
#include "tc/core/cuda/cuda_backend.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/tensor.h"
#include "tc/library/group_convolution.h"

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
  auto N = FLAGS_N;
  auto G = FLAGS_G;
  auto C = FLAGS_C;
  auto F = FLAGS_F;
  auto H = FLAGS_H;
  auto W = FLAGS_W;
  auto KH = FLAGS_KH;
  auto KW = FLAGS_KW;
  auto threads = (W >= 10) ? std::vector<size_t>{W / 4, H / 2}
                           : std::vector<size_t>{4, 8, 4};
  auto options = tc::CudaMappingOptions::makeNaiveMappingOptions()
                     .tile(1, 1, 1)
                     .mapToThreads(threads)
                     .mapToBlocks({32, 32})
                     .useSharedMemory(true)
                     .usePrivateMemory(false)
                     .unroll(1);

  std::vector<int64_t> I_sizes{N, G, C, H, W};
  std::vector<int64_t> W1_sizes{G, F, C, KH, KW};
  std::vector<int64_t> B_sizes{G, F};
  DLDataType floatType{DLDataTypeCode::kDLFloat, 32, 1};

  auto I = makeDLConstTensor(tc::TensorInfo{
      floatType, 32, I_sizes, tc::makeStridesFromSizes(I_sizes)});
  auto W1 = makeDLConstTensor(tc::TensorInfo{
      floatType, 32, W1_sizes, tc::makeStridesFromSizes(W1_sizes)});
  auto B = makeDLConstTensor(tc::TensorInfo{
      floatType, 32, B_sizes, tc::makeStridesFromSizes(B_sizes)});

  auto res = tc::compile<tc::CudaBackend>(
      tc::makeGroupConvolution2DTc(1, 1),
      "group_convolution",
      {I.get(), W1.get(), B.get()},
      options);

  std::cout << res.source << std::endl;
}
