set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(group_convolution_sample group_convolution.cc)
target_link_libraries(group_convolution_sample tc_autotuner tc_core tc_aten ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})
target_include_directories(group_convolution_sample PUBLIC ../tc)
