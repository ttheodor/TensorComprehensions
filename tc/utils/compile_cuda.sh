#!/bin/zsh

s=$(($(./retriever --size)-1))

seq 0 ${s} | parallel 'nvcc --gpu-architecture=compute_60  --use_fast_math -std=c++11 -x cu -c =(./retriever --idx={}) -o {}.o'
