#include "tc/aot/common.h"

#include <random>
#include <vector>

#include <boost/functional/hash.hpp>
#include <gflags/gflags.h>

#include "tc/version/version.h"

namespace {
uint64_t getMaxSize(const std::vector<tc::TensorInfo>& ti) {
  std::vector<int64_t> ms;
  for (const auto& t : ti) {
    ms.push_back(*std::max_element(t.shape.begin(), t.shape.end()));
  }
  return *std::max_element(ms.begin(), ms.end());
}
} // namespace

namespace tc {

OptionsGenerator::OptionsGenerator(const std::vector<tc::TensorInfo>& ti)
    : maxSize{getMaxSize(ti)},
      rng{pcg_extras::seed_seq_from<std::random_device>{}} {}

tc::CudaMappingOptions OptionsGenerator::operator()() const {
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
                     .useSharedMemory(makeBool())
                     .useReadOnlyCache(makeBool())
                     .matchLibraryCalls(false);
  options.unrollCopyShared(
      options.proto().use_shared_memory() ? makeBool() : false);

  return options;
};

tc::FusionStrategy OptionsGenerator::makeFusionStrategy() const {
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

std::vector<uint64_t> OptionsGenerator::makeTiles() const {
  std::vector<uint64_t> sizes(3);
  sizes[0] = 1;
  sizes[1] = 1;
  sizes[2] = std::uniform_int_distribution<uint64_t>{0, maxSize}(rng);

  return sizes;
}

uint64_t OptionsGenerator::oneToMaxSize() const {
  return std::uniform_int_distribution<uint64_t>{1, maxSize}(rng);
}
std::vector<uint64_t> OptionsGenerator::makeCudaDim() const {
  std::vector<uint64_t> sizes(3);
  std::generate_n(sizes.begin(), 3, [this]() { return oneToMaxSize(); });
  return sizes;
}

std::vector<uint64_t> OptionsGenerator::makeBlock() const {
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

std::vector<uint64_t> OptionsGenerator::makeGrid() const {
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

bool OptionsGenerator::makeBool() const {
  return std::uniform_int_distribution<int>{0, 1}(rng);
}

uint64_t OptionsGenerator::makeUnroll() const {
  return oneToMaxSize();
}

GCInputsGenerator::GCInputsGenerator()
    : rng{pcg_extras::seed_seq_from<std::random_device>{}} {}

WaveNetInputsGenerator::WaveNetInputsGenerator()
    : rng{pcg_extras::seed_seq_from<std::random_device>{}} {}

namespace {
DEFINE_int64(KHW_low, 1, "Kernel Height/Width lower bound");
DEFINE_int64(KHW_high, 9, "Kernel Height/Width upper bound");
DEFINE_int64(HW_low, 8, "Image Height/Width lower bound");
DEFINE_int64(HW_high, 64, "Image Height/Width upper bound");
DEFINE_int64(CF_low, 4, "");
DEFINE_int64(CF_high, 32, "");
} // namespace

std::vector<tc::TensorInfo> GCInputsGenerator::operator()() const {
  auto KHW = std::uniform_int_distribution<int64_t>{FLAGS_KHW_low,
                                                    FLAGS_KHW_high}(rng);
  auto HW =
      std::uniform_int_distribution<int64_t>{FLAGS_HW_low, FLAGS_HW_high}(rng);
  auto CF =
      std::uniform_int_distribution<int64_t>{FLAGS_CF_low, FLAGS_CF_high}(rng);

  std::vector<int64_t> I_sizes{32, 32, CF, HW, HW};
  std::vector<int64_t> W1_sizes{32, CF, CF, KHW, KHW};
  std::vector<int64_t> B_sizes{32, CF};
  DLDataType floatType{DLDataTypeCode::kDLFloat, 32, 1};

  return {
      tc::TensorInfo{floatType, 32, I_sizes, tc::makeStridesFromSizes(I_sizes)},
      tc::TensorInfo{
          floatType, 32, W1_sizes, tc::makeStridesFromSizes(W1_sizes)},
      tc::TensorInfo{
          floatType, 32, B_sizes, tc::makeStridesFromSizes(B_sizes)}};
}

namespace {
DEFINE_uint32(B_low, 1, "Batch size");
DEFINE_uint32(B_high, 32, "Batch size");
DEFINE_uint32(
    RESIDUAL_C_low,
    1,
    "Residual channels (i.e. WaveNet block input channels)");
DEFINE_uint32(
    RESIDUAL_C_high,
    64,
    "Residual channels (i.e. WaveNet block input channels)");
DEFINE_uint32(
    DILATION_C_low,
    1,
    "Dilation channels (i.e. WaveNet block channels after dilated convolution)");
DEFINE_uint32(
    DILATION_C_high,
    64,
    "Dilation channels (i.e. WaveNet block channels after dilated convolution)");
DEFINE_uint32(
    SKIP_C_low,
    1,
    "Skip channels (i.e. WaveNet block channels in the skip tensor)");
DEFINE_uint32(
    SKIP_C_high,
    64,
    "Skip channels (i.e. WaveNet block channels in the skip tensor)");
DEFINE_uint32(
    RECEPTIVE_FIELD,
    4000,
    "https://arxiv.org/pdf/1609.03499.pdf paper mentions 16K samples per second"
    "and a receptive field of 240ms so we approx. set the default to 4000)");
DEFINE_uint32(DILATION_FACTOR_low, 0, "Powers of 2 from 1 to 512 in the paper");
DEFINE_uint32(
    DILATION_FACTOR_high,
    9,
    "Powers of 2 from 1 to 512 in the paper");
} // namespace

std::vector<tc::TensorInfo> WaveNetInputsGenerator::operator()() const {
  auto B =
      std::uniform_int_distribution<int64_t>{FLAGS_B_low, FLAGS_B_high}(rng);
  auto RESIDUAL_C = std::uniform_int_distribution<int64_t>{
      FLAGS_RESIDUAL_C_low, FLAGS_RESIDUAL_C_high}(rng);
  auto DILATION_C = std::uniform_int_distribution<int64_t>{
      FLAGS_DILATION_C_low, FLAGS_RESIDUAL_C_high}(rng);
  auto SKIP_C = std::uniform_int_distribution<int64_t>{FLAGS_SKIP_C_low,
                                                       FLAGS_SKIP_C_high}(rng);
  auto DILATION_FACTOR = 1l
      << std::uniform_int_distribution<int64_t>{FLAGS_DILATION_FACTOR_low,
                                                FLAGS_DILATION_FACTOR_low}(rng);

  std::vector<int64_t> Data{B, RESIDUAL_C, FLAGS_RECEPTIVE_FIELD};
  std::vector<int64_t> FilterWeight{DILATION_C, RESIDUAL_C, 2};
  std::vector<int64_t> FilterBias{DILATION_C};
  std::vector<int64_t> GateWeight{DILATION_C, RESIDUAL_C, 2};
  std::vector<int64_t> GateBias{DILATION_C};
  std::vector<int64_t> ResWeight{RESIDUAL_C, DILATION_C};
  std::vector<int64_t> ResBias{RESIDUAL_C};
  std::vector<int64_t> SkipWeight{SKIP_C, DILATION_C};
  std::vector<int64_t> SkipBias{SKIP_C};
  std::vector<int64_t> Dilation{DILATION_FACTOR};

  DLDataType floatType{DLDataTypeCode::kDLFloat, 32, 1};

  return {
      tc::TensorInfo{floatType, 32, Data, tc::makeStridesFromSizes(Data)},
      tc::TensorInfo{
          floatType, 32, FilterWeight, tc::makeStridesFromSizes(FilterWeight)},
      tc::TensorInfo{
          floatType, 32, FilterBias, tc::makeStridesFromSizes(FilterBias)},
      tc::TensorInfo{
          floatType, 32, GateWeight, tc::makeStridesFromSizes(GateWeight)},
      tc::TensorInfo{
          floatType, 32, GateBias, tc::makeStridesFromSizes(GateBias)},
      tc::TensorInfo{
          floatType, 32, ResWeight, tc::makeStridesFromSizes(ResWeight)},
      tc::TensorInfo{floatType, 32, ResBias, tc::makeStridesFromSizes(ResBias)},
      tc::TensorInfo{
          floatType, 32, SkipWeight, tc::makeStridesFromSizes(SkipWeight)},
      tc::TensorInfo{
          floatType, 32, SkipBias, tc::makeStridesFromSizes(SkipBias)},
      tc::TensorInfo{
          floatType, 32, Dilation, tc::makeStridesFromSizes(Dilation)}};
}

tc::KernelInfo makeKernelInfo(
    const tc::CudaCompilationResult& res,
    uint64_t id,
    const std::string& tc,
    const std::vector<tc::TensorInfo>& inputsInfo,
    const std::vector<tc::TensorInfo>& outputsInfo,
    const tc::CudaMappingOptions& opts,
    const std::chrono::high_resolution_clock::duration& compilation_time) {
  tc::KernelInfo ki;
  ki.set_id(id);
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
  return ki;
}

std::size_t hash_value(const tc::TensorInfo& ti) {
  size_t seed = 0;
  boost::hash_combine(seed, ti.dtype.bits);
  boost::hash_combine(seed, ti.dtype.code);
  boost::hash_combine(seed, ti.dtype.lanes);
  boost::hash_combine(seed, ti.alignment);
  for (auto i : ti.shape)
    boost::hash_combine(seed, i);
  for (auto i : ti.strides)
    boost::hash_combine(seed, i);

  return seed;
}

size_t TensorInfoHash::operator()(
    const std::vector<tc::TensorInfo>& tis) const {
  size_t seed = 0;
  for (const auto& ti : tis) {
    boost::hash_combine(seed, ti);
  }
  return seed;
}
} // namespace tc
