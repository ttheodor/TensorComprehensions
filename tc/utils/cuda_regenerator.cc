#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sys/stat.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string_view>

#include "tc/core/cuda/cuda_compilation_cache.h"
#include "tc/core/cuda/cuda_tc_executor.h"

DEFINE_string(cache, "", "Serialized Protobuf");

using namespace tc;
namespace fs = std::filesystem;

void write_proto(const OptionsCache& cache, std::string_view fname, int idx) {
  fs::path prefix = "/tmp/caches";
  if (fs::exists(prefix) and not fs::is_directory(prefix)) {
    std::stringstream ss;
    ss << prefix << " exists but it's not a directory" << std::endl;
    throw std::runtime_error{ss.str()};
  } else {
    fs::create_directory(prefix);
  }
  auto f = fs::path(fname).filename();
  auto output = prefix;
  output /= f.stem();
  output += std::to_string(idx++);
  output.replace_extension(f.extension());

  std::fstream serialized(
      output.string(), std::ios::binary | std::ios::trunc | std::ios::out);
  if (!serialized) {
    std::cout << "Failed to open the output stream for dumping protobuf: "
              << output;
  } else {
    cache.toProtobuf().SerializePartialToOstream(&serialized);
  }
}

std::string TC = R"(
def group_convolution(float(N,G,C,H,W) I, float(G,F,C,KH,KW) W1, float(G,F) B)
-> (O)
{
    O(n, g, f, h, w) +=!
        I(n, g, r_c, h + r_kh, w + r_kw) * W1(g, f, r_c, r_kh, r_kw)
    O(n, g, f, h, w)  = O(n, g, f, h, w) + B(g, f)
}
)";

void write_tc(OptionsCache& cache) {
  for (auto& entry : cache) {
    entry.key.id = TC;
  }
}
std::vector<const DLTensor*> toConstDlpackTensors(
    const std::vector<tc::detail::TensorInfo>& tensors) {
  std::vector<const DLTensor*> dlTensors;
  for (auto tensor : tensors) {
    // LEAK EVERYTHING
    auto t = new DLTensor;
    t->dtype = tensor.dType;
    t->shape = new int64_t[tensor.shape.size()];
    t->ndim = tensor.shape.size();
    std::copy(tensor.shape.begin(), tensor.shape.end(), t->shape);
    if (tensor.strides.empty()) {
      t->strides = nullptr;
    } else {
      t->strides = new int64_t[tensor.strides.size()];
      std::copy(tensor.strides.begin(), tensor.strides.end(), t->strides);
    }
    dlTensors.push_back(t);
  }
  return dlTensors;
}

lang::TreeRef parseDefs(const std::string& language) {
  lang::Parser parser(language);
  std::vector<lang::TreeRef> res;
  while (parser.L.cur().kind != lang::TK_EOF) {
    res.push_back(parser.parseFunction());
  }
  if (res.size() != 1) {
    throw std::invalid_argument{"More than one TC in language."};
  }
  return res.front();
}

void generate_cuda(OptionsCache& cache) {
  std::vector<std::future<void>> jobs;
  for (auto& entry : cache) {
    for (auto& v : entry.values) {
      jobs.push_back(std::async([&]() {
        auto inputs = toConstDlpackTensors(entry.key.inputs);
        auto lang = parseDefs(TC);
        CudaTcExecutor exec{"group_convolution",
                            inputs,
                            v.mappingOptions.toProtobufSerializedString(),
                            lang};
        exec.compile(v.mappingOptions);
        v.cuda_source = exec.cudaSource;
      }));
    }
  }

  while (not jobs.empty()) {
    std::cout << "\rGenerating CUDA  " << jobs.size() << " jobs unfinished."
              << std::flush;
    jobs.erase(
        std::remove_if(
            jobs.begin(),
            jobs.end(),
            [](const auto& job) {
              return std::future_status::ready ==
                  job.wait_for(std::chrono::milliseconds{10});
            }),
        jobs.end());
  }
}

int main(int argc, char* argv[]) {
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);
  auto fnames = FLAGS_cache;
  auto n = std::count(fnames.begin(), fnames.end(), ',') + 1;
  std::replace(fnames.begin(), fnames.end(), ',', ' ');
  std::istringstream ss(fnames);

  auto readBuf = [](const std::string& filename) {
    OptionsCacheProto buf;
    struct stat buffer = {0};
    if (stat(filename.c_str(), &buffer) == 0) {
      std::ifstream serialized(filename, std::ios::binary);
      if (not buf.ParseFromIstream(&serialized)) {
        throw std::runtime_error{"Could not parse proto from " + filename};
      }
    }
    return buf;
  };

  int idx = 0;
  for (auto fname = std::istream_iterator<std::string>(ss);
       fname != std::istream_iterator<std::string>();
       ++fname) {
    std::cout << "Finished cache " << idx << '/' << n << std::endl;
    try {
      OptionsCache cache(readBuf(*fname));
      if (cache.totalSize() == 0) {
        std::cout << *fname << " is empty" << std::endl;
        continue;
      }
      write_tc(cache);
      generate_cuda(cache);
      write_proto(cache, *fname, idx++);
    } catch (std::exception& e) {
      std::cout << e.what() << std::endl;
    }
  }
}
