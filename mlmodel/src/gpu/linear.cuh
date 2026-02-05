#ifndef LINEAR_CUH
#define LINEAR_CUH

#include "utils.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__device__ inline void linearForward(float* X, float* W, float* B, float* Z, const int M, const int N, const int K, const int D){
    
    const int tid = threadIdx.x;
    const int row = blockIdx.z;
    const int col = tid * 4;
    if(D == 1){
        if(row < M && blockIdx.x == 0 && blockIdx.y == 0){
           
            if(col<N){
                reinterpret_cast<float4*>(&Z[row * N + col])[0] = reinterpret_cast<const float4*>(&B[col])[0];
            }
            
            __syncthreads();
        }
        await();
    }
    

    sgemm<BM, BN, BK, TM, TN>(X, W, Z, 1.0f, 1.0f, M, N, K, D);
    
   
}

__device__ inline void simpleLinear(float* X, float* W, float* B, float* Z, const int M, const int N, const int K){
    const int tid = threadIdx.x;
    const int col = blockIdx.x;
    const int vecPerRow = K / 4;   
    const int mask = (1u << vecPerRow) - 1;


    if (tid >= vecPerRow || col >= N) return;
    alignas(16)float regX[4];
    reinterpret_cast<float4*>(regX)[0] = reinterpret_cast<float4*>(&X[tid * 4])[0];

    alignas(16) float regW[4];
    reinterpret_cast<float4*>(regW)[0] = reinterpret_cast<float4*>(&W[col * K + tid * 4])[0];

    float regZ = 0.0f;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        regZ += regX[i] * regW[i];
    }

    #pragma unroll
    for(int offset = vecPerRow / 2; offset > 0; offset >>= 1){
            regZ += __shfl_down_sync(mask, regZ, offset);   
        }

        
    if(tid == 0){
        Z[col] = regZ + B[col];
    }
}




template<const int BM, const int BN, const int BK, const int TM, const int TN, const int TK>
__device__ inline void linearBackward(float* dL_dZ, float* dL_dW, float* dL_dX, float* X, float* W,
     const int M, const int N, const int K, const int D){
   
    
    sgemm_naive(X, dL_dZ, dL_dW, 1.0f, 0.0f, K, N, M, D);

    sgemm<BM, BK, BN, TM, TK>(dL_dZ, W, dL_dX, 1.0f, 0.0f, M, K, N, D);
}

__device__ inline void simpleLinearBackward(float* dL_dZ, float* dL_dW, float* dL_dX, float* X, float* W,
     const int M, const int N, const int K){
    
    sgemm_naive(X, dL_dZ, dL_dW, 1.0f, 0.0f, K, N, M, 1);

    sgemm_naive(dL_dZ, W, dL_dX, 1.0f, 0.0f, M, K, N, 1);
}

#endif 