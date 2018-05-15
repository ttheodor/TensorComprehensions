add_executable(random_sample random_sample.cc)
target_link_libraries(random_sample PUBLIC tc_cuda tc_aten ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES} stdc++fs)
set_property(TARGET random_sample PROPERTY CXX_STANDARD 17)
