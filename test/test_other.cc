/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <iostream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "tc/core/cuda/cuda_mapping_options.h"
#include "tc/core/polyhedral/cuda/mapped_scop.h"
#include "tc/core/polyhedral/scop.h"
#include "tc/lang/parser.h"

using namespace std;

using namespace tc;
using namespace tc::polyhedral;
using namespace tc::polyhedral::detail;

namespace {
DEFINE_uint64(N, 2048, "");
DEFINE_uint64(M, 1024, "");
DEFINE_uint64(T0, 1, "");
DEFINE_uint64(T1, 1, "");
DEFINE_uint64(B0, 1, "");
DEFINE_uint64(B1, 1, "");
DEFINE_uint64(G0, 1, "");
DEFINE_uint64(G1, 1, "");
} // namespace

int main(int argc, char** argv) {
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);

  string tc = R"TC(
def fun(float(N, M) A, float(N, M) B) -> (C) {
    C(n, m) = A(n, m) + B(n, m)
}
)TC";
  lang::Parser p(tc);
  auto tcTree = p.parseFunction();

  // lang::TreeRef tcTree(tc);
  auto halideComponents =
      tc2halide::translate(isl::with_exceptions::globalIslCtx(), tcTree);

  auto scopTmp = polyhedral::Scop::makeScop(
      isl::with_exceptions::globalIslCtx(), halideComponents);

  auto context = scopTmp->makeContext(
      std::unordered_map<std::string, int>{{"N", FLAGS_N}, {"M", FLAGS_M}});

  scopTmp = polyhedral::Scop::makeSpecializedScop(
      *scopTmp, context.intersect(scopTmp->globalParameterContext));

  auto opts = CudaMappingOptions::makePointwiseCudaMappingOptions()
                  .tile(FLAGS_T0, FLAGS_T1)
                  .mapToThreads({FLAGS_B0, FLAGS_B1})
                  .mapToBlocks({FLAGS_G0, FLAGS_G1})
                  .unroll(0);
  auto mappedScop =
      polyhedral::MappedScop::makeWithOuterBlockInnerThreadStrategy(
          std::move(scopTmp), opts);

  std::cout << mappedScop->scop() << std::endl;

  auto res = mappedScop->codegen("foo");
  std::cout << "\n\n" << std::get<0>(res) << std::endl;
}
