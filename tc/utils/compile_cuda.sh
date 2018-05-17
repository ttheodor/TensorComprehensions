#!/bin/bash




retriever=${1}
protofile=${2}
benchmaker_header_dir=${3}
export benchmaker_header_dir

com="${retriever} --input=${protofile} "

export com

s_1=$(($(${com} --size) - 1)) 

 
function pp {
    local id=${1}
    c="${com} --idx=${id}"


    local source=$(${c})
    local specialized_name=$(${c} --sname)
    local params=$(${c} --params)
    local noutputs=$(${c} --noutputs)
    local ninputs=$(${c} --ninputs)
    local id=$(${c} --id)
    local grid=$(${c} --grid)
    local block=$(${c} --block)

    inputs="static_cast<const float32*>(inputs[0])"

    for i in `seq 1 $((${ninputs}-1))`; do
        inputs="${inputs}, static_cast<const float32*>(inputs[${i}])"
    done

    outputs="static_cast<float32*>(outputs[0])"
    for o in `seq 1 $((${noutputs}-1))`; do
        outputs="${outputs}, static_cast<float32*>(outputs[${o}])"
    done

    printf "#include \"benchmark_register.hpp\"
namespace{
${source}

auto dummy = [](){
  Register::get().registerBenchmark([]
  (const std::vector<const void*>& inputs, std::vector<void*>& outputs){
    dim3 grid{${grid}};
    dim3 block{${block}};
    auto t0 = std::chrono::high_resolution_clock::now();
      ${specialized_name}<<<grid, block>>>(${params}, ${outputs}, ${inputs}); 
    auto t1 = std::chrono::high_resolution_clock::now();
    return t1 - t0;}, ${id});
    return 0;
}();

}\n"
}

export -f pp

function compile {
    cuda=$(pp ${1})
    fname=$(mktemp /tmp/cuda_sourceXXXXXX.cu)
    printf "${cuda}\n" > ${fname}
    nvcc ${fname} -I${benchmaker_header_dir} -c -O3 --use_fast_math -std=c++14 -o cuda_${1}.o 
}

export -f compile

seq 0 ${s_1} | parallel -a - compile {}
