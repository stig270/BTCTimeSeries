#include "utils.cuh"
#include "linear.cuh"
#include "activationfunctions.cuh"

#ifndef SPARSEMOE_CUH
#define SPARSEMOE_CUH

struct SparseMoeBuffers {
    float* gate;
    float* probs;
    float* sorted;
    float* indices;
    float* experts;
    float* out;
};

struct SparseMoeParams {
    float* gw; 
    float* gb; 
    float* w; 
    float* b; 
};



template<const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__device__ inline void sparseMoeForward(float* X, const SparseMoeBuffers& buffers, const SparseMoeParams params, int T, int C, int E, int K){
   
    linearForward<BM, BN, BK, TM, TN>(X, params.gw, params.gb, buffers.gate, T, E, C, 1);
    softmaxForward(buffers.gate, buffers.probs, T, E);
    mergesortForward(buffers.probs, buffers.sorted, buffers.indices, T, E);


}

#endif