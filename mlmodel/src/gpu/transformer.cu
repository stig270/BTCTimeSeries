#include "utils.cuh"
#include "activationfunctions.cuh"
#include "linear.cuh"
#include "sparsemoe.cuh"
#include "rmsnorm.cuh"
#include "attention.cuh"
#include "optimizer.cuh"
#include "config.h"
#include <cuda_runtime.h>

#define ceildiv(a, b) ((a + b - 1) / b)


template<const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void classifierForward(float* X, float* XNorm, float* WNorm, float* BNorm, float* ZNorm, float* W, float* B, float* Z, float* Y, int C, int F, float epsilon){
    RMSNormForward<BM, BN, BK, TM, TN>(X, XNorm, WNorm, BNorm, ZNorm, 1, C, epsilon);
    simpleLinear(ZNorm, W, B, Z, 1.0f, 1.0f, 1, F, C, 1); // optimised sgemm is overkill for what is essentially 2 dot products
    simpleSoftmaxForward(Z, Y);
}

template<const uint BM, const uint BN, const uint BK, const uint TM, const uint TN >
__global__ void flashAttentionForward(float* X, float* attnNorm, float* WNorm, float* BNorm, float* attnX, float* Y, const AttentionBuffers& buffers,
    const AttentionParams& params, int T, int C, int N, int H, float epsilon){
    RMSNormForward<BM, BN, BK, TM, TN>(X, attnNorm, WNorm, BNorm, attnX, T, C, epsilon);
    attentionForward<BM, BN, BK, TM, TN>(attnX, buffers, params, T, C, N, H);
    int row = blockIdx.z;
    int tid = threadIdx.x;

    float regX[4];
    float regY[4];
    if(tid < (C / 4) && row < T){
        reinterpret_cast<float4*>(regX)[0] = reinterpret_cast<float4*>(&X[row * C + tid * 4])[0];
        reinterpret_cast<float4*>(regY)[0] = reinterpret_cast<float4*>(&buffers.projY)[0];
        __syncthreads();

        #pragma unroll
        for(int i = 0; i < 4; ++i){
            regY[i] += regX[i];
        }
        reinterpret_cast<float4*>(&Y[row * C + tid * 4])[0] = regY;
    }
    

}


void transformerForward(float* X, float* Y, const ModelParameters& params, const ModelHyperparameters& config, IntermediateBuffers& buffers){

    const int C = config.C;
    const int T = config.T;
    const int N = config.N;
    const int H = config.H;
    const int E = config.E;
    const int L = config.L;
    const int F = config.F;
    const int K = config.K;
    const float epsilon = config.epsilon;

    uint BM, BN, BK, TM, TN; 

    for(int l = 0; l < L; ++l){
        BM = 16;
        BN = 16;
        BK = 8;
        TM = 4;
        TN = 4;

        dim3 gridDim(N, ceildiv(3 * C, BN), T);
        dim3 blockDim((BM * BN)/(TM * TN), 1, 1);
        
        AttentionBuffers attnBuffers{
            buffers.qkv[l],
            buffers.Q[l],
            buffers.K[l],
            buffers.V[l],
            buffers.attnScores[l],
            buffers.W[l],
            buffers.attnY[l],
            buffers.projX[l],
            buffers.projY[l]
        };

        AttentionParams attnParams{
            params.qkvw[l],
            params.qkvb[l],
            params.projW[l],
            params.projB[l]
        };

        flashAttentionForward<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(
            buffers.attnNormX[l], buffers.attnNorm[l], params.WNorm[2*l], params.BNorm[2*l],
            buffers.attnX[l], attnBuffers, attnParams, T, C, N, H, epsilon
        );



            //sparsemoeforward...




    }
    


    BM = 1; 
    BN = 32; 
    BK = 8; 
    TM = 1; 
    TN = 4; 

    dim3 gridDim(C, 1, ceildiv(C, BN));
    dim3 blockDim((BM * BN)/(TM * TN), 1, 1);

    classifierForward<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(
        buffers.moeOut2[2 * L], buffers.CLSNorm, params.WNorm[2*L + 1], params.BNorm[2*L + 1],
        buffers.ZNormCLS, params.WCLS, params.BCLS, buffers.ZCLS, Y, C, F, epsilon
    );
    


}