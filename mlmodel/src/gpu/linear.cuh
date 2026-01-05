#include "utils.cuh"
#include <cuda_runtime.h>
#ifndef LINEAR_CUH
#define LINEAR_CUH


template<const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__device__ inline void linearForward(float* X, float* W, float* B, float* Z, int M, int N, int K, int D){
    const uint tid = threadIdx.x;
    const uint vecPerRow = N / 4;
    const uint row = tid / vecPerRow;
    const uint col = (tid % vecPerRow) * 4;
    const uint offset = blockIdx.x * M * N;

    if(tid < M * N && D = 1){
        reinterpret_cast<float4*>(&Z[offset + row * N + col])[0] = reinterpret_cast<const float4*>(&B[offset + col])[0];
    }
    __syncthreads();
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

template<const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__device__ inline void linearBackward(float* dL_dZ, float* dL_dW, float* dL_dX, float* X, float* W, int M, int N, int K, int D){
   
    
    sgemm<BM, BN, BK, TM, TN>(X, dL_dZ, dL_dW, 1.0f, 0.0f, K, N, M, D);

    __syncthreads();

    sgemm<BM, BN, BK, TM, TN>(dL_dZ, W, dL_dX, 1.0f, 0.0f, M, K, N, D);
}

__device__ inline void simpleLinearBackward(float* dL_dZ, float* dL_dW, float* dL_dX, float* X, float* W, int M, int N, int K){
    
    sgemm_naive(X, dL_dZ, dL_dW, 1.0f, 0.0f, K, N, M, 1);

    __syncthreads();

    sgemm_naive(dL_dZ, W, dL_dX, 1.0f, 0.0f, M, K, N, 1);
}