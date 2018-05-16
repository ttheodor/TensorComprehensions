#include <bitset>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>

#include <gflags/gflags.h>

#include "tc/core/cuda/cuda_mapping_options_cpp_printer.h"
#include "tc/proto/aot.pb.h"

namespace {
DEFINE_string(input, "kernels.proto", "input filename (default: kernels.proto");
DEFINE_bool(size, false, "print number of kernels");
DEFINE_uint64(idx, 0, "Choose kernel [0, size-1] (default: 0)");
DEFINE_bool(block, false, "print the block size");
DEFINE_bool(grid, false, "print the grid size");
DEFINE_bool(params, false, "print params");
DEFINE_bool(options, false, "print the mapping options");
DEFINE_bool(id, false, "print the id");
DEFINE_bool(ninputs, false, "print the number of inputs");
DEFINE_bool(noutputs, false, "print the number of ouputs");
DEFINE_bool(sname, false, "print the specialized kernel name");
} // namespace

bool ispowerof2(unsigned int x) {
  return x && !(x & (x - 1));
}

bool moreThanOneSet(std::vector<bool> b) {
  auto s = std::accumulate(b.begin(), b.end(), 0);
  return s > 1;
}

int main(int argc, char* argv[]) {
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (not std::filesystem::exists(FLAGS_input)) {
    std::cout << FLAGS_input << " does not exists" << std::endl;
    return 1;
  }

  tc::AotBuf kernels;
  {
    std::ifstream input{FLAGS_input, std::ios::binary};
    if (not kernels.ParseFromIstream(&input)) {
      std::cout << FLAGS_input << " does not contain a valid protobuf."
                << std::endl;
      return 2;
    }
  }
  if (kernels.kernels_size() == 0) {
    std::cout << "The loaded protobuf is empty." << std::endl;
    return 0;
  }

  if (FLAGS_size) {
    std::cout << kernels.kernels_size() << std::endl;
    return 0;
  }

  if (FLAGS_idx >= static_cast<uint64_t>(kernels.kernels_size())) {
    std::cout << "idx out of range" << std::endl;
    return 3;
  }

  if (moreThanOneSet({FLAGS_block,
                      FLAGS_grid,
                      FLAGS_options,
                      FLAGS_params,
                      FLAGS_id,
                      FLAGS_ninputs,
                      FLAGS_noutputs,
                      FLAGS_sname})) {
    std::cout << "Either specify one options or none (to get the Cuda source)."
              << std::endl;
    return 4;
  }
  const auto& kernel = kernels.kernels(FLAGS_idx);

  auto cudaDimToString = [](const auto& b) {
    std::stringstream ss;
    ss << b.x() << ',' << (b.has_y() ? b.y() : 1ul) << ','
       << (b.has_z() ? b.z() : 1ul);
    return ss.str();
  };

  if (FLAGS_block) {
    std::cout << cudaDimToString(kernel.tight_block()) << std::endl;
    return 0;
  }

  if (FLAGS_grid) {
    std::cout << cudaDimToString(kernel.tight_grid()) << std::endl;
    return 0;
  }

  if (FLAGS_options) {
    std::cout << tc::CudaMappingOptionsAsCpp{tc::CudaMappingOptions{
                     kernel.kernel_options()}}
              << std::endl;
    return 0;
  }

  if (FLAGS_params) {
    std::copy(
        kernel.parameters().begin(),
        kernel.parameters().end() - 1,
        std::ostream_iterator<int64_t>{std::cout, ","});
    std::cout << *(kernel.parameters().end() - 1) << std::endl;
    return 0;
  }

  if (FLAGS_ninputs) {
    std::cout << kernel.inputs_size() << std::endl;
    return 0;
  }

  if (FLAGS_noutputs) {
    std::cout << kernel.outputs_size() << std::endl;
    return 0;
  }

  if (FLAGS_sname) {
    std::cout << kernel.specialized_name() << std::endl;
    return 0;
  }

  if (FLAGS_id) {
    std::cout << kernel.id() << std::endl;
    return 0;
  }

  std::cout << kernels.kernels(FLAGS_idx).cuda_source() << std::endl;
  return 0;
}
