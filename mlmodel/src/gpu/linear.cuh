#include "utils.cuh"
#include <cuda_runtime.h>
#ifndef LINEAR_CUH
#define LINEAR_CUH


template<const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__device__ inline void linearForward(float* X, float* W, float* B, float* Z, int M, int N, int K, int D){
    if(D == 1){ //basic linear case, also works for attn
        const uint tid = threadIdx.x;
        if(tid < M * N && blockIdx.x == 0){
            const uint vecPerRow = N / 4;
            const uint row = tid / vecPerRow;
            const uint col = (tid % vecPerRow) * 4;
            reinterpret_cast<float4*>(&Z[row * N + col])[0] = reinterpret_cast<const float4*>(&B[col])[0];
        }
    }
    else{// moe case
        
    }
    sgemm<BM, BN, BK, TM, TN>(X, W, Z, 1.0f, 1.0f, M, N, K, D);
   
}

__device__ inline void simpleLinear(float* X, float* W, float* B, float* Z, int M, int N, int K){
    const uint tid = threadIdx.x;
    const uint col = blockIdx.x;
    const uint vecPerRow = K / 4;   
    const uint mask = (1u << vecPerRow) - 1;


    if (tid >= vecPerRow) return;

    float regX[4];
    reinterpret_cast<float4*>(regX)[0] = reinterpret_cast<float4*>(&X[tid * 4])[0];

    float regW[4];
    reinterpret_cast<float4*>(regW)[0] = reinterpret_cast<float4*>(&W[col * K + tid * 4])[0];

    float regZ = 0.0f;

    #pragma unroll
    for (uint i = 0; i < 4; ++i) {
        regZ += regX[i] * regW[i];
    }

    #pragma unroll
    for(uint offset = vecPerRow / 2; offset > 0; offset >>= 1){
            regZ += __shfl_down_sync(mask, regZ, offset);   
        }

        
    if(tid == 0){
        Z[col] = regZ + B[col];
    }
     
}


#endif

