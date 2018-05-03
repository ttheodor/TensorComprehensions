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
DEFINE_uint64(N, 32, "");
DEFINE_uint64(G, 32, "");
DEFINE_uint64(C, 4, "");
DEFINE_uint64(F, 4, "");
DEFINE_uint64(W, 56, "");
DEFINE_uint64(H, 56, "");
DEFINE_uint64(KW, 3, "");
DEFINE_uint64(KH, 3, "");
DEFINE_uint64(T0, 0, "");
DEFINE_uint64(T1, 0, "");
DEFINE_uint64(T2, 0, "");
DEFINE_uint64(T3, 0, "");
DEFINE_uint64(T4, 0, "");
DEFINE_uint64(T5, 0, "");
DEFINE_uint64(T6, 0, "");
DEFINE_uint64(B0, 1, "");
DEFINE_uint64(B1, 1, "");
DEFINE_uint64(B2, 1, "");
DEFINE_uint64(G0, 1, "");
DEFINE_uint64(G1, 1, "");
DEFINE_uint64(G2, 1, "");
} // namespace

int main(int argc, char** argv) {
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::google::InitGoogleLogging(argv[0]);

  string tc = R"TC(
def group_convolution(float(N,G,C,H,W) I, float(G,F,C,KH,KW) W1, float(G,F) B)
-> (O)
{
    O(n, g, f, h, w) +=!
        I(n, g, r_c, h + r_kh, w + r_kw) * W1(g, f, r_c, r_kh, r_kw)
    O(n, g, f, h, w)  = O(n, g, f, h, w) + B(g, f)
}
)TC";
  lang::Parser p(tc);
  auto tcTree = p.parseFunction();

  // lang::TreeRef tcTree(tc);
  auto halideComponents =
      tc2halide::translate(isl::with_exceptions::globalIslCtx(), tcTree);

  auto scopTmp = polyhedral::Scop::makeScop(
      isl::with_exceptions::globalIslCtx(), halideComponents);

  // auto context = scopTmp->makeContext(
  // std::unordered_map<std::string, int>{{"N", FLAGS_N},
  //{"G", FLAGS_G},
  //{"C", FLAGS_C},
  //{"F", FLAGS_F},
  //{"W", FLAGS_W},
  //{"H", FLAGS_H},
  //{"KW", FLAGS_KW},
  //{"KH", FLAGS_KH}});
  auto context = scopTmp->makeContext(std::unordered_map<std::string, int>{});

  scopTmp = polyhedral::Scop::makeSpecializedScop(
      *scopTmp, context.intersect(scopTmp->globalParameterContext));

  auto opts = CudaMappingOptions::makePointwiseCudaMappingOptions()
                  .tile(
                      FLAGS_T0,
                      FLAGS_T1,
                      FLAGS_T2,
                      FLAGS_T3,
                      FLAGS_T4,
                      FLAGS_T5,
                      FLAGS_T6)
                  .mapToThreads({FLAGS_B0, FLAGS_B1, FLAGS_B2})
                  .mapToBlocks({FLAGS_G0, FLAGS_G1, FLAGS_G2})
                  .unroll(0)
                  .useSharedMemory(true);
  auto mappedScop =
      polyhedral::MappedScop::makeWithOuterBlockInnerThreadStrategy(
          std::move(scopTmp), opts);

  std::cout << mappedScop->scop() << std::endl;

  auto res = mappedScop->codegen("foo");
  std::cout << "\n\n" << std::get<0>(res) << std::endl;
}
