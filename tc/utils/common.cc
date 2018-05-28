#include "tc/utils/common.h"

#include <boost/functional/hash.hpp>
#include <random>

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
                     .useSharedMemory(makeBool());
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

std::vector<tc::TensorInfo> GCInputsGenerator::operator()() const {
  auto KHW = std::uniform_int_distribution<int64_t>{1, 9}(rng);
  auto HW = std::uniform_int_distribution<int64_t>{9, 64}(rng);
  auto CF = std::uniform_int_distribution<int64_t>{4, 32}(rng);

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

OptionsAndInputsGenerator::OptionsAndInputsGenerator(
    uint64_t number_inputs,
    uint64_t number_options)
    : number_inputs{number_inputs}, number_options{number_options} {
  GCInputsGenerator ig;
  do {
    data[ig.operator()()];
  } while (data.size() < number_inputs);
}

std::pair<std::vector<tc::TensorInfo>, CudaMappingOptions>
OptionsAndInputsGenerator::generate() {
  std::lock_guard<std::mutex> lock{mtx};
  for (auto& [inputs, options] : data) {
    if (options.size() >= number_options)
      continue;
    OptionsGenerator og{inputs};

    while (true) {
      auto opts = og();
      if (options.count(opts) > 0)
        continue;
      options.insert(opts);
      return std::make_pair(inputs, opts);
    }
  }

  throw std::runtime_error{"Enough requested pairs have been generated."};
}

void OptionsAndInputsGenerator::remove(
    const std::vector<tc::TensorInfo>& inputs,
    const CudaMappingOptions& options) {
  data[inputs].erase(options);
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
