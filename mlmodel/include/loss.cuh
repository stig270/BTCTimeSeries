#ifndef LOSS_CUH
#define LOSS_CUH

#include "config.h"
#include "utils.cuh"
#include <cuda_runtime.h>

__global__ void binaryCrossEntropyLossForward(float* Y, float* P, float* loss,  int F, float epsilon){
   int tid = threadIdx.x;
   if(tid < F){
       float y = Y[tid];
       float p = P[tid];
       float l = - (y * logf(p + epsilon) + (1 - y) * logf(1 - p + epsilon));
       atomicAdd(loss, l);
   }
   

}

__device__ inline void binaryCrossEntropyLossBackward(float* Y, float* P, float* dL_dP, int F, float epsilon){
    int tid = threadIdx.x;
    if(tid < F){
        float y = Y[tid];
        float p = P[tid];
        dL_dP[tid] = - (y / (p + epsilon)) + ((1 - y) / (1 - p + epsilon));
    }
}

// much simpler than BCE with softmax Jacobian
__device__ inline void BCESoftmaxBackward(float* Y, float* P, float* dL_dZ, int F){
    int tid = threadIdx.x;
    if(tid < F){
        float y = Y[tid];
        float p = P[tid];
        dL_dZ[tid] = F * (p - y);
    }
}


__device__ inline void loadBalance(float* loadBalance, int* indices, int T, int E, int K){
    const int token = blockIdx.x;

    if(token < T){

        float counts[4] = {0.0f};

        #pragma unroll  
        for(int i = 0; i<K ; ++i){
            counts[indices[token * K + i]] += 1.0f;
        }

        #pragma unroll
        for(int i = 0; i<E; ++i){
            atomicAdd(&loadBalance[i], counts[i] / T);
        }
    }
}   
#endif
