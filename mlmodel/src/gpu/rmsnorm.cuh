#ifndef RMSNORM_CUH
#define RMSNORM_CUH
#include "utils.cuh"
#include "linear.cuh"
#include <cuda_runtime.h>

template<const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__device__ inline void RMSNormForward(float* X, float* XNorm, float* W, float* B, float* Z, float* RMS, int T, int C, float epsilon){

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

        RMS[row] = rms;

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

template<const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__device__ inline void RMSNormBackward(float* dL_dZ, float* dL_dXNorm, float* dL_dW, float* dL_dX, 
    float* XNorm,float* W, float* RMS, int C, float epsilon){

    linearBackward<BM, BN, BK, TM, TN>(
        dL_dZ,
        XNorm,
        W,
        dL_dXNorm,
        dL_dW,
        T,
        C,
        C,
        1
    );

    float yg;
    dotProduct(dL_dXNorm, XNorm, &yg, C);

    const uint tid = threadIdx.x;
    float regG[4];
    float regXNorm[4];
    if(tid < C / 4){
        reinterpret_cast<float4*>(regG)[0] = reinterpret_cast<const float4*>(&dL_dXNorm[tid * 4])[0];
        reinterpret_cast<float4*>(regXNorm)[0] = reinterpret_cast<const float4*>(&XNorm[tid * 4])[0];
        float dL_dX[4]; 
        #pragma unroll
        for(int i = 0; i < 4; i ++){
            dL_dX[i] = (regG[i] - (yg * regXNorm[i]) / C) * RMS[0];
            
        }
        reinterpret_cast<float4*>(&dL_dX[tid * 4])[0] = reinterpret_cast<const float4*>(dL_dX)[0];
    }
}
#endif
