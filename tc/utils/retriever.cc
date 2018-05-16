#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include <gflags/gflags.h>

#include "tc/proto/aot.pb.h"

namespace {
DEFINE_string(input, "kernels.proto", "input filename (default: kernels.proto");
DEFINE_bool(size, false, "print number of kernels");
DEFINE_uint64(idx, 0, "Choose kernel [0, size-1] (default: 0)");
DEFINE_bool(block, false, "print the block size");
DEFINE_bool(grid, false, "print the grid size");
} // namespace

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

  if (FLAGS_block and FLAGS_grid) {
    std::cout << "Use either block, grid, or none. Not both." << std::endl;
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

  std::cout << kernels.kernels(FLAGS_idx).cuda_source() << std::endl;
  return 0;
}
