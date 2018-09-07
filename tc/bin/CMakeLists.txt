add_executable(tc2cl tc2cl.cc)
target_link_libraries(tc2cl PUBLIC tc_core tc_code_opencl)
set_property(TARGET tc2cl PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
