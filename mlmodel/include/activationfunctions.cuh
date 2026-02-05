#ifndef ACTIVATIONFUNCTIONS_CUH
#define ACTIVATIONFUNCTIONS_CUH

#include "utils.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

__device__ inline void softmaxForward(float* X, float* Y, int C, int T, int N){
    const int tid = threadIdx.x;
    const int vecPerRow = C / 4;
    const int baseOffset = (blockIdx.x * T + blockIdx.z) * C;
    const int mask = (1u << vecPerRow) - 1;


    if(blockIdx.x < N && blockIdx.z < T && blockIdx.y == 0){
        if(tid < vecPerRow){
            float regX[4];
            
            reinterpret_cast<float4*>(regX)[0] = reinterpret_cast<const float4*>(&X[baseOffset + tid * 4])[0];
            

            float threadMax = fmaxf(fmaxf(regX[0], regX[1]), fmaxf(regX[2], regX[3]));

            #pragma unroll
            for (int offset = vecPerRow / 2; offset > 0; offset >>= 1) {
                threadMax = fmaxf(threadMax, __shfl_down_sync(mask, threadMax, offset));
            }
            // optionally for larger vectors, could include warp level max. This covers up to 128 elements per row so is sufficient for this project.
            float maxVal = __shfl_sync(mask, threadMax, 0);
        
            
            float threadSum = 0.0f; 
            #pragma unroll
            for(int i = 0; i < 4; ++i){
                regX[i] = expf(regX[i] - maxVal);
                threadSum += regX[i];
            }
            
            
            #pragma unroll
            for(int offset = vecPerRow / 2; offset > 0; offset >>= 1){
                threadSum += __shfl_down_sync(mask, threadSum, offset);
            }

            threadSum = __shfl_sync(mask, threadSum, 0);

            
            #pragma unroll
            for(int i = 0; i < 4; ++i){
                regX[i] /= threadSum;
            }
            
            reinterpret_cast<float4*>(&Y[baseOffset + tid * 4])[0] = reinterpret_cast<float4*>(regX)[0];

        }
        __syncthreads();    
    }
    await();
    
    
}

__device__ inline void softmaxBackward(float* dL_dY, float* Y, float* J,  float* dL_dX, int C, int T, int N){
    
    const int head = blockIdx.x;
    const int row = blockIdx.z;
    const int tid = threadIdx.x;
    const int offset = (head * T + row) * C;

    dotProduct(Y, dL_dY, J, C, T, N); 
    if(tid < (C / 4) && head < N && row < T){
        float regdL_dY[4];
        float regY[4];
        float regdL_dX[4] = {0.0f};
        reinterpret_cast<float4*>(regdL_dY)[0] = reinterpret_cast<float4*>(&dL_dY[offset + tid * 4])[0];
        reinterpret_cast<float4*>(regY)[0] = reinterpret_cast<float4*>(&Y[offset + tid * 4])[0];
        #pragma unroll
        for(int i = 0; i < 4; ++i){
            regdL_dX[i] = regY[i] * (regdL_dY[i] - J[(offset)]);
        }

        reinterpret_cast<float4*>(&dL_dX[offset + tid * 4])[0] = reinterpret_cast<float4*>(regdL_dX)[0];
    }
}



__device__ inline void simpleSoftmaxForward(float* X, float* Y){
    
    const int tid = threadIdx.x;
    if (tid != 0 || blockIdx.x != 0) return;

    alignas(8) float regX[2] = {0.0f};
    reinterpret_cast<float2*>(regX)[0] = reinterpret_cast<const float2*>(&X[0])[0];

    float maxVal = fmaxf(regX[0], regX[1]);
    regX[0] = expf(regX[0] - maxVal);
    regX[1] = expf(regX[1] - maxVal);
    float sumVal = regX[0] + regX[1];
    regX[0] /= sumVal;
    regX[1] /= sumVal;

    reinterpret_cast<float2*>(&Y[0])[0] = reinterpret_cast<float2*>(regX)[0];
    // simpler version for the classifier, the other version is overkill and would not work.  
}

__device__ inline float siluForward(float x){
    return x / (1 + expf(-x));
}

__device__ inline float siluBackward(float x){
    float sigmoid = 1.0f / (1.0f + expf(-x));
    return sigmoid * (1 + x * (1 - sigmoid));
}





#endif