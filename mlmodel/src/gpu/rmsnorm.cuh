#include "utils.cuh"
#include "linear.cuh"
#include <cuda_runtime.h>
#ifndef RMSNORM_CUH
#define RMSNORM_CUH


template<const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__device__ inline void RMSNormForward(float* X, float* XNorm, float* W, float* B, float* Z, int T, int C, float epsilon){

    const uint vecPerRow = C / 4;
    const uint row = blockidx.z;
    const uint tid = threadidx.x;
    const uint mask = (1u << vecPerRow) - 1;

    if (tid < vecPerRow && row < T) {
            
        float regX[4];

        const float* Xrow = &X[row * C];

        
        reinterpret_cast<float4*>(regX)[0] = reinterpret_cast<const float4*>(&Xrow[tid * 4])[0];
        


        const uint warpSize = 32;
        const uint laneid = tid % warpSize;

        float sum = 0.0f;
        
        #pragma unroll
        for(uint i = 0; i < 4; i ++){
            sum += regX[i] * regX[i];
        }
        

        #pragma unroll
        for(uint offset = vecPerRow / 2; offset > 0; offset >>= 1){
            sum += __shfl_down_sync(mask, sum, offset);
        }

        float rms;
        if(laneid == 0){
            rms = rsqrtf(epsilon + sum / C);
        }

        rms = __shfl_sync(mask, rms, 0);

        
        #pragma unroll
        for(uint i = 0; i < 4; i ++){
            regX[i] *= rms;
        }
        

        float* XNormRow = &XNorm[row * C];
        
        
        reinterpret_cast<float4*>(&XNormRow[tid * 4])[0] = reinterpret_cast<const float4*>(regX)[0];
        

        linearForward<BM, BN, BK, TM, TN>(
            XNorm,
            W,
            B,
            Z,
            T,
            C,
            C,
            1
        );

    }


}




#endif
