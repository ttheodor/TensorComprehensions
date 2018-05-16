add_executable(gc_generator group_convolution_generator.cc)
target_link_libraries(gc_generator PUBLIC tc_core pcg stdc++fs)
set_property(TARGET gc_generator PROPERTY CXX_STANDARD 17)

add_executable(retriever retriever.cc)
target_link_libraries(retriever PUBLIC tc_proto stdc++fs ${GFLAGS_LIBRARIES})
set_property(TARGET retriever PROPERTY CXX_STANDARD 17)
