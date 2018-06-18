#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include <pcg_random.hpp>

#include "tc/core/cuda/cuda_backend.h"
#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/tensor.h"
#include "tc/proto/aot.pb.h"

namespace tc {

class OptionsGenerator {
 public:
  OptionsGenerator(const std::vector<tc::TensorInfo>& ti);

  tc::CudaMappingOptions operator()() const;

 private:
  tc::FusionStrategy makeFusionStrategy() const;
  std::vector<uint64_t> makeTiles() const;
  uint64_t oneToMaxSize() const;
  std::vector<uint64_t> makeCudaDim() const;
  std::vector<uint64_t> makeBlock() const;
  std::vector<uint64_t> makeGrid() const;
  bool makeBool() const;
  uint64_t makeUnroll() const;

  uint64_t maxSize;
  mutable pcg64 rng;
};

class GCInputsGenerator {
 public:
  GCInputsGenerator();
  std::vector<tc::TensorInfo> operator()() const;

 private:
  mutable pcg64 rng;
};

tc::KernelInfo makeKernelInfo(
    const tc::CudaCompilationResult& res,
    uint64_t id,
    const std::string& tc,
    const std::vector<tc::TensorInfo>& inputsInfo,
    const std::vector<tc::TensorInfo>& outputsInfo,
    const tc::CudaMappingOptions& opts,
    const std::chrono::high_resolution_clock::duration& compilation_time);

struct OptionsHash {
  size_t operator()(const tc::CudaMappingOptions& o) const {
    return std::hash<std::string>{}(o.proto().SerializeAsString());
  }
};

struct TensorInfoHash {
  size_t operator()(const std::vector<tc::TensorInfo>& tis) const;
};

class OptionsAndInputsGenerator {
 public:
  OptionsAndInputsGenerator(uint64_t number_inputs, uint64_t number_options);

  std::pair<std::vector<tc::TensorInfo>, CudaMappingOptions> generate();
  void remove(
      const std::vector<tc::TensorInfo>& inputs,
      const CudaMappingOptions& options);

 private:
  uint64_t number_inputs;
  uint64_t number_options;
  std::mutex mtx;
  std::unordered_map<
      std::vector<tc::TensorInfo>,
      std::unordered_set<tc::CudaMappingOptions, OptionsHash>,
      TensorInfoHash>
      data;
};

} // namespace tc
