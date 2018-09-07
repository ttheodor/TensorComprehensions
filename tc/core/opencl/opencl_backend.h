/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <string>
#include <vector>

#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/cuda/cuda_mapping_options_cpp_printer.h"
#include "tc/core/halide_utils.h"
#include "tc/core/tensor.h"

namespace tc {
/**
 * Information returned by polyhedral compilation. In particular, since we use
 * tightening of loop bounds for CUDA kernels, it includes the actual grid and
 * block sizes needed at runtime.
 */
struct OpenCLCompilationResult {
  std::string source;
  std::string specializedName;
  std::vector<long> parameters;
  Grid grid;
  Block block;
};

struct OpenCLTcExecutor;

/**
 * This type declares the dependent types and static functions needed to
 * autotune, compile and run for the CPU backend.
 */
struct OpenCLBackend {
  using ExecutorType = OpenCLTcExecutor;
  using MappingOptionsType = CudaMappingOptions;
  using CompilationResultType = OpenCLCompilationResult;

  /// Main entry point for polyhedral compilation
  static CompilationResultType compileWithTcMapper(
      const std::string& tcName,
      tc2halide::HalideComponents halideComponents,
      const std::vector<const DLConstTensor*>& inputs,
      /* TODO: in the future also pass outputs for stride and alignment */
      const MappingOptionsType& options);
};
} // namespace tc
