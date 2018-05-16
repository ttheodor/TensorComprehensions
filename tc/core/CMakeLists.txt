################################################################################
# tc_core
#
# Core CPU library with cross-compilation capabilities linked from
# tc_core_cuda
################################################################################
add_library(
  tc_core

  SHARED

  flags.cc
  mapping_options.cc
  mapping_options_cpp_printer.cc
  islpp.cc
  compiler.cc
  tensor.cc

  tc2halide.cc

  halide2isl.cc
  halide_utils.cc

  polyhedral/codegen.cc
  polyhedral/memory_promotion.cc
  polyhedral/reduction_matcher.cc
  polyhedral/schedule_isl_conversion.cc
  polyhedral/schedule_transforms.cc
  polyhedral/schedule_tree.cc
  polyhedral/schedule_tree_elem.cc
  polyhedral/schedule_print.cc
  polyhedral/scop.cc
  polyhedral/separation.cc
  polyhedral/unroll.cc
  cuda/cuda_backend.cc
  cuda/cuda_mapping_options.cc
  cuda/cuda_mapping_options_cpp_printer.cc
  polyhedral/cuda/codegen.cc
  polyhedral/cuda/mapped_scop.cc
  polyhedral/cuda/mapping_types.cc
  polyhedral/cuda/memory_promotion_heuristic.cc
  polyhedral/cuda/tighten_launch_bounds.cc

)
target_link_libraries(
  tc_core

  ${HALIDE_LIBRARIES}
  ${ISL_LIBRARIES}
  ${GFLAGS_LIBRARIES} 
  ${GLOG_LIBRARIES}
  tc_lang
  tc_version
  tc_proto
)
install(
  TARGETS
  tc_core

  DESTINATION lib
)
