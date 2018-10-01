#include <fstream>
#include <iostream>
#include <iterator>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "tc/core/compiler.h"
#include "tc/core/opencl/opencl_backend.h"
#include "tc/core/tensor.h"

namespace {
DEFINE_string(
    sizes,
    "",
    "Comma separated list of input sizes, e.g., --sizes=\"N=14,K=3,M=55\".");
}

std::string readInputFile(const std::string& filename) {
  std::ifstream input{filename};
  return {std::istreambuf_iterator<char>{input},
          std::istreambuf_iterator<char>()};
}

class Tensor {
 public:
  Tensor(
      const lang::Param& p,
      const std::unordered_map<std::string, uint64_t>& sizeMap)
      : name_{p.ident().name()} {
    for (const auto& dim : p.tensorType().dims()) {
      TC_CHECK_EQ(dim->kind(), lang::TK_IDENT);
      std::string dname = lang::Ident(dim).name();

      TC_CHECK(sizeMap.count(dname) == 1ul)
          << "Size " << dname << " not specified.";
      sizes.push_back(sizeMap.at(dname));
    }
  }

  const std::string& name() const {
    return name_;
  }

  friend std::ostream& operator<<(std::ostream& out, const Tensor& t) {
    out << t.name_ << '[';

    for (uint64_t i = 0; i < t.sizes.size() - 1; ++i)
      out << t.sizes[i] << " , ";

    if (not t.sizes.empty())
      out << t.sizes.back();

    out << ']';
    return out;
  }

  const DLConstTensor* toDLT() const {
    // XXX: leak everything and let the OS do the dirty work
    DLConstTensor* t = new DLConstTensor();
    // float: type_code = 2, bits = 32, lanes=1
    t->dtype = {2, 32, 1};
    t->ndim = sizes.size();
    t->shape = new int64_t[t->ndim];
    std::copy(sizes.begin(), sizes.end(), t->shape);
    t->strides = new int64_t[t->ndim];
    auto strides = tc::makeStridesFromSizes(sizes);
    std::copy(strides.begin(), strides.end(), t->strides);
    t->byte_offset = 0;
    return t;
  };

 private:
  std::string name_;
  std::vector<uint64_t> sizes;
};

std::vector<Tensor> makeTensors(
    const lang::TreeRef& t,
    const std::unordered_map<std::string, uint64_t>& sizeMap) {
  std::vector<Tensor> ts;
  for (auto p : lang::Def(t).params()) {
    ts.push_back(Tensor(p, sizeMap));
  }
  return ts;
}

std::vector<const DLConstTensor*> makeInputs(const std::vector<Tensor>& ts) {
  std::vector<const DLConstTensor*> inputs;
  std::transform(
      ts.begin(), ts.end(), std::back_inserter(inputs), [](const Tensor& t) {
        return t.toDLT();
      });

  return inputs;
}

std::vector<std::string> split(const std::string& s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

std::unordered_map<std::string, uint64_t> parseSizes() {
  std::unordered_map<std::string, uint64_t> sizes;
  for (const auto& token : split(FLAGS_sizes, ',')) {
    auto stokens = split(token, '=');
    TC_CHECK_EQ(stokens.size(), 2ul)
        << "The following size specification is invalid: " << stokens;
    TC_CHECK_EQ(sizes.count(stokens[0]), 0ul)
        << "Size " << stokens[0] << " specified more than once.";
    sizes[stokens[0]] = std::stoul(stokens[1]);
  }
  return sizes;
}

std::pair<std::string, std::vector<tc::TensorInfo>> generateOpenCL(
    const lang::TreeRef& t,
    const std::vector<const DLConstTensor*> inputs) {
  auto inputsInfo = tc::makeTensorInfoVector(inputs);
  auto outputsInfo = tc::detail::inferOutputTensorInfo(t, inputs, {});
  auto halideComponents =
      tc2halide::translate(isl::with_exceptions::globalIslCtx(), t, {});
  tc::detail::checkInputsCompliant(halideComponents, inputs);

  auto tcName = lang::Def(t).name().name();
  auto compilationResult = tc::OpenCLBackend::compileWithTcMapper(
      tcName,
      halideComponents,
      inputs,
      /* TODO outputs, */
      tc::CudaMappingOptions::makeNaiveMappingOptions()
          .mapToThreads({1, 1, 1})
          .mapToBlocks({1, 1, 1}));

  return {compilationResult.source, outputsInfo};
}

int main(int argc, char* argv[]) {
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);

  if (argc != 2) {
    std::cerr << "Input filename missing" << std::endl;
    exit(-1);
  }

  auto parsedTCs = tc::detail::parse(readInputFile(argv[1]));
  auto inputSizes = parseSizes();

  for (const auto& p : parsedTCs) {
    const auto& entryPoint = p.first;
    const auto& TC = p.second;
    std::cout << "//Generating code for " << entryPoint << std::endl;
    auto ts = makeTensors(TC, inputSizes);
    auto res = generateOpenCL(TC, makeInputs(ts));
    std::cout << res.first;
    for (const auto& t : ts)
      std::cout << "// " << t << std::endl;

    int i = 0;
    for (const auto& o : res.second) {
      std::cout << "// Output" << i++ << '[';
      for (uint64_t d = 0; d < o.shape.size() - 1; ++d) {
        std::cout << o.shape[d] << " , ";
      }
      if (not o.shape.empty() > 0)
        std::cout << o.shape.back();
      std::cout << ']' << std::endl;
    }
  }
}
