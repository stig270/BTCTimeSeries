#ifndef RMSNORM_CUH
#define RMSNORM_CUH
#include "utils.cuh"
#include "linear.cuh"
#include <cuda_runtime.h>


template<const int BM, const int BN, const int BK, const int TM, const int TN>
__device__ inline void RMSNormForward(float* X, float* XNorm, float* W, float* B, float* Z, float* RMS, const int T, const int C, const float epsilon){

    const int vecPerRow = C / 4;
    const int row = blockIdx.z;
    const int tid = threadIdx.x;
    const int mask = (1u << vecPerRow) - 1;

    if (row < T && blockIdx.x == 0 && blockIdx.y == 0){
            

        if(tid<vecPerRow){
            alignas(16) float regX[4] = {0.0f};


            
            reinterpret_cast<float4*>(regX)[0] = reinterpret_cast<const float4*>(&X[row * C + tid * 4])[0];
            
            float sum = 0.0f;
            
            #pragma unroll
            for(int i = 0; i < 4; i ++){
                sum += regX[i] * regX[i];
            }
            

            #pragma unroll
            for(int offset = vecPerRow / 2; offset > 0; offset >>= 1){
                sum += __shfl_down_sync(mask, sum, offset);
            }


            float rms;
            if(tid == 0){
                rms = rsqrtf(epsilon + sum / C);
            }


            rms = __shfl_sync(mask, rms, 0);

            if(row == T-1){
                RMS[0] = rms;
            }

            #pragma unroll
            for(int i = 0; i < 4; i ++){
                regX[i] *= rms;
            }
            

            
            
            reinterpret_cast<float4*>(&XNorm[row * C + tid * 4])[0] = reinterpret_cast<const float4*>(regX)[0];
        }
        __syncthreads();

    }
    await();
    linearForward<BM, BN, BK, TM, TN>(XNorm, W, B, Z, T, C, C, 1);


}

template<const int BM, const int BN, const int BK, const int TM, const int TN, const int TK>
__device__ inline void RMSNormBackward(float* dL_dZ, float* dL_dXNorm, float* dL_dW, float* dL_dX, 
    float* XNorm,float* W, float RMS, const int C, const float epsilon){

    linearBackward<BM, BN, BK, TM, TN, TK>(dL_dZ, dL_dW, dL_dXNorm, XNorm, W, 1, C, C, 1);

    float yg;
    dotProduct(dL_dXNorm, XNorm, &yg, C, 1, 1);

    const int tid = threadIdx.x;
    
    if(tid < C / 4 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0){
        alignas(16) float regG[4];
        alignas(16) float regXNorm[4];
        reinterpret_cast<float4*>(regG)[0] = reinterpret_cast<const float4*>(&dL_dXNorm[tid * 4])[0];
        reinterpret_cast<float4*>(regXNorm)[0] = reinterpret_cast<const float4*>(&XNorm[tid * 4])[0];
        alignas(16) float dL_dX[4]; 
        #pragma unroll
        for(int i = 0; i < 4; i ++){
            dL_dX[i] = (regG[i] - (yg * regXNorm[i]) / C) * RMS;
            
        }
        reinterpret_cast<float4*>(&dL_dX[tid * 4])[0] = reinterpret_cast<const float4*>(dL_dX)[0];
    }
    __syncthreads();
    await();
}
#endif
