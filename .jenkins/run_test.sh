#!/bin/bash

set -ex

source /etc/lsb-release

# condition: if 14.04 and conda, also run python tests
# condition: if 16.04 and conda, run only python tests
# condition: if any and non-conda, run test.sh only

# TODO: modify 2LUT tests from example_MLP_model and enable on CI

echo "Running TC backend tests"
FILTER_OUT=MLP_model ./test.sh
./build/tc/benchmarks/MLP_model --gtest_filter=-*2LUT*
if [[ $(conda --version | wc -c) -ne 0 ]]; then
  source activate tc-env
  echo "Running TC PyTorch tests"
  ./test_python/run_test.sh
fi
