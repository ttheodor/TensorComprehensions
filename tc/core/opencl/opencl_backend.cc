#include "tc/core/opencl/opencl_backend.h"

#include "tc/core/polyhedral/opencl/mapped_scop.h"
#include "tc/core/polyhedral/scop.h"

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

OpenCLBackend::CompilationResultType OpenCLBackend::compileWithTcMapper(
    const std::string& tcName,
    tc2halide::HalideComponents halideComponents,
    const std::vector<const DLConstTensor*>& inputs,
    /* TODO: in the future also pass outputs for stride and alignment */
    const MappingOptionsType& options) {
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
      polyhedral::opencl::MappedScop::makeWithOuterBlockInnerThreadStrategy(
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
  std::tie(source, grid, block) = mappedScop->codegen(specializedName);
  LOG_IF(INFO, FLAGS_dump_cuda) << "generatedCuda: " << source << "\n"
                                << "grid: " << grid << " block: " << block;

  return OpenCLCompilationResult{
      source, specializedName, parameters, grid, block};
}

} // namespace tc
