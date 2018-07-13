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

#include "tc/core/check.h"
#include "tc/core/cuda/cuda_backend.h"
#include "tc/core/cuda/cuda_mapping_options_cpp_printer.h"
#include "tc/core/halide_utils.h"
#include "tc/core/polyhedral/cuda/mapped_scop.h"
#include "tc/core/tc2halide.h"
#include "tc/core/tensor.h"

#include "tc/lang/parser.h"
#include "tc/lang/sema.h"

#include <utility>

namespace tc {
namespace {
// Append ordered values to the kernel name, separated by "_".
template <typename T>
std::string specializeKernelName(
    const std::string& tcName,
    std::vector<T> params) {
  std::stringstream ss;
  ss << tcName;
  for (auto i : params) {
    ss << "_" << i;
  }
  return ss.str();
}
} // namespace

CudaCompilationResult CudaBackend::compileWithTcMapper(
    const std::string& tcName,
    tc2halide::HalideComponents halideComponents,
    const std::vector<const DLConstTensor*>& inputs,
    /* TODO: in the future also pass outputs for stride and alignment info */
    const CudaMappingOptions& options,
    bool dropExternC) {
  // A bit chicken-and-eggy, need scop from TC to have the space to build the
  // context to specialize the scop..
  auto scop = polyhedral::Scop::makeScop(
      isl::with_exceptions::globalIslCtx(), halideComponents);
  auto pvm = computeParamValueMap(halideComponents, inputs);
  scop = polyhedral::Scop::makeSpecializedScop(*scop, pvm);
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << options;
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << "original schedule:\n"
                                      << *(scop->scheduleRoot());

  // Now we can build stuff
  auto mappedScop =
      polyhedral::MappedScop::makeWithOuterBlockInnerThreadStrategy(
          std::move(scop), options);
  LOG_IF(INFO, FLAGS_debug_tc_mapper) << "Mapped schedule:" << std::endl
                                      << *(mappedScop->schedule());

  auto parameters = mappedScop->scop().getParameterValues();
  auto specializedName = specializeKernelName(tcName, parameters);

  // This updates the launch bounds with the actual result from compilation
  // with tightening of launch_bounds. What you get is not necessarily what
  // you asked for, the autotuner should adapt to that.
  std::string source;
  Grid grid;
  Block block;
  std::tie(source, grid, block) =
      mappedScop->codegen(specializedName, dropExternC);
  LOG_IF(INFO, FLAGS_dump_cuda) << "generatedCuda: " << source << "\n"
                                << "grid: " << grid << " block: " << block;

  return CudaCompilationResult{
      source, specializedName, parameters, grid, block};
}
} // namespace tc
