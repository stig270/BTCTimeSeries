#include "utils.cuh"
#include "linear.cuh"
#include "activationfunctions.cuh"

#ifndef SPARSEMOE_CUH
#define SPARSEMOE_CUH

struct SparseMoeBuffers {
    float* gate;
    float* probs;
    float* sorted;
    int* indices;
    float* X;
    float* W;
    float* B;
    float* experts;
    float* Y;
};

struct SparseMoeParams {
    float* gw; 
    float* gb; 
    float* w; 
    float* b; 
};



__device__ inline void sparseMoeForward(float* X, const SparseMoeBuffers& buffers, const SparseMoeParams& params, int T, int Cin, int Cout, int E, int K){
    
    float* gate = buffers.gate;
    float* probs = buffers.probs;
    float* sorted = buffers.sorted;
    int* indices = buffers.indices;
    float* W = buffers.W;
    float* B = buffers.B;
    float* experts = buffers.experts;
    float* Y = buffers.Y;
    float* sum = buffers.sum;
    float* gw = params.gw;
    float* gb = params.gb;
    float* w = params.w;
    float* b = params.b;

   
    linearForward<Cin, 2, 8, 2, 1>(X, gw, gb, gate, T, E, Cin, 1);
    softmaxForward(gate, probs, T, E);
    mergesortForward(probs, sorted, indices, T, E);
    
    const uint tid = 4 * threadIdx.x;
    const uint token = blockIdx.x;
    const uint row = blockIdx.y;
    uint expertIdx;
    if(token < T && row < Cin){
        #pragma unroll
        for(int i = 0; i<K; ++i){
            expertIdx = indices[token * E + i];

            reinterpret_cast<float4*>(W[token * Cin * K * Cout + row * K * Cout + i * Cout + tid])[0] = reinterpret_cast<float4*>(&W[expertIdx * Cin * Cout + row * Cout + tid])[0];
            reinterpret_cast<float4*>(B[token * K * Cout + i * Cout * tid])[0] = reinterpret_cast<float4*>(&B[expertIdx * Cout + tid])[0];
            reinterpret_cast<float4*>(experts[token * K * Cout + i * Cout * tid])[0] = reinterpret_cast<float4*>(&B[token * K * Cout + i * Cout * tid])[0];
        } // double check this when GPU available 
    }

    linearForward<1, 32, 32, 1, 2>(X, W, B, experts,  1,  2 * Cout, Cin, T);
        
    __shared__ float sortedShared[T * E];

    if(tid < E && token < T){
        reinterpret_cast<float4*>(sortedShared[token * E + tid])[0] = reinterpret_cast<float4*>(&sorted[token * E + tid])[0];
    
    #pragma unroll

    for(int i = 0; i < K; ++i){
            float4 regY = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 regExpert = reinterpret_cast<float4*>(&experts[token * K * Cout + i * Cout + tid])[0];
            float regProb = sortedShared[token * E + i];

            #pragma unroll
            for(int j = 0; j < 4; ++j){
                regY[j] += regExpert[j] * regProb;
            }

            reinterpret_cast<float4*>(&Y[token * Cout + tid])[0] = regY;
        }
    }
        __syncthreads();


    
}

#endif