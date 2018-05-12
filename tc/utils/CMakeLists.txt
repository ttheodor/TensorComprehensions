add_executable(tc_cuda_regenerator cuda_regenerator.cc)
target_link_libraries(tc_cuda_regenerator PUBLIC tc_cuda tc_proto ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES} stdc++fs)
install(TARGETS tc_cuda_regenerator DESTINATION bin)
set_property(TARGET tc_cuda_regenerator PROPERTY CXX_STANDARD 17)
