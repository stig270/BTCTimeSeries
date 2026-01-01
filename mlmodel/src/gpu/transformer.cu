#include "utils.cuh"
#include "activationfunctions.cuh"
#include "linear.cuh"
#include "sparsemoe.cuh"
#include "rmsnorm.cuh"
#include "attention.cuh"
#include "optimizer.cuh"
#include "loss.cuh"
#include "config.h"
#include <cuda_runtime.h>

#define ceildiv(a, b) ((a + b - 1) / b)


__global__ void calculateTheta(float* X, float* theta, int T, int H, int B){
    int token = blockIdx.x;
    int tid = threadIdx.x;
    if(token > T && tid > H / 2) return;
    theta[token * H/2 + tid] = token * powf(B,  2 * (float)tid / H);
}

__global__ void classifierForward(float* X, float* XNorm, float* WNorm, float* BNorm, float* ZNorm, float* W, float* B, float* Z, float* Y, int C, int F, float epsilon){
    RMSNormForward<1, 32, 8, 1, 4>(X, XNorm, WNorm, BNorm, ZNorm, 1, C, epsilon);
    simpleLinear(ZNorm, W, B, Z, 1, F, C); // optimised sgemm is overkill for what is essentially 2 dot products
    simpleSoftmaxForward(Z, Y);
}

__global__ void classifierBackward(float* Y, float* P, float* dL_dZCLS, int F, float epsilon){
    BCESoftmaxBackward(Y, P, dL_dZCLS, F);

}

__global__ void flashAttentionForward(float* X, float* attnNorm, float* WNorm, float* BNorm, float* attnX, float* Y, const AttentionBuffers& buffers,
    const AttentionParams& params, int T, int C, int N, int H, float epsilon){
    RMSNormForward<16, 16, 8, 4, 4>(X, attnNorm, WNorm, BNorm, attnX, T, C, epsilon);
    attentionForward(attnX, buffers, params, T, C, N, H);
    int row = blockIdx.z;
    int tid = threadIdx.x;

    float regX[4];
    float regY[4];
    if(tid < (C / 4) && row < T){
        reinterpret_cast<float4*>(regX)[0] = reinterpret_cast<float4*>(&X[row * C + tid * 4])[0];
        reinterpret_cast<float4*>(regY)[0] = reinterpret_cast<float4*>(&buffers.projY[row * C + tid * 4])[0];
        __syncthreads();

        #pragma unroll
        for(int i = 0; i < 4; ++i){
            regY[i] += regX[i];
        }
        reinterpret_cast<float4*>(&Y[row * C + tid * 4])[0] = regY;
    }
    

}


__global__ void fusedSparsemoeForward(float* X, float* Y, float* moeNorm, float* moeHidden, float* WNorm, float* BNorm, 
    const SparseMoeBuffers& buffers1, const SparseMoeParams& params1, const SparseMoeBuffers& buffers2,
    const SparseMoeParams& params2, int T, int C, int E, int K, float epsilon){
    RMSNormForward<16, 16, 8, 4, 4>(X, moeNorm, WNorm, BNorm, moeX, 1, C, epsilon);
    sparseMoeForward(moeX, buffers1, params1, T, C, C * 4, E, K);
    int row = blockIdx.z;
    int tid = threadIdx.x;

    float regX[4];
    float regY[4] = {0.0f};
    if(tid < (C / 4) && row < T){
        reinterpret_cast<float4*>(regX)[0] = reinterpret_cast<float4*>(&X[row * C + tid * 4])[0];
        __syncthreads();

        #pragma unroll
        for(int i = 0; i < 4; ++i){
            regY[i] = siluForward(regX[i]);
        }
        reinterpret_cast<float4*>(&moeHidden[row * C + tid * 4])[0] = regY;
    }
    sparseMoeForward(moeHidden, buffers2, params2, T, C * 4, C, E, K);

    float regX[4];
    float regY[4];
    if(tid < (C / 4) && row < T){
        reinterpret_cast<float4*>(regX)[0] = reinterpret_cast<float4*>(&X[row * C + tid * 4])[0];
        reinterpret_cast<float4*>(regY)[0] = reinterpret_cast<float4*>(&buffers2.moeOut[row * C + tid * 4])[0];
        __syncthreads();

        #pragma unroll
        for(int i = 0; i < 4; ++i){
            regY[i] += regX[i];
        }
        reinterpret_cast<float4*>(&Y[row * C + tid * 4])[0] = regY;
    }
}

void calculateThetaLauncher(float* X, float* theta, int T, int H, int B){
    dim3 gridDim(T, 1, 1);
    dim3 blockDim(H / 2, 1, 1);

    calculateTheta<<<gridDim, blockDim>>>(X, theta, T, H, B);
}

void transformerForward(ModelParameters& params, const ModelHyperparameters& config, IntermediateBuffers& buffers){

    const int C = config.C;
    const int T = config.T;
    const int N = config.N;
    const int H = config.H;
    const int E = config.E;
    const int L = config.L;
    const int F = config.F;
    const int K = config.K;
    const float epsilon = config.epsilon;


    for(int l = 0; l < L; ++l){
       

        dim3 attnGrid(N, C, T);
        dim3 attnBlock(C, 1, 1);  

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

        flashAttentionForward<<<attnGrid, attnBlock>>>(
            buffers.attnNormX[l], buffers.attnNorm[l], params.WNorm[2*l], params.BNorm[2*l],
            buffers.attnX[l],buffers.moeHiddenX1[l], attnBuffers, attnParams, T, C, N, H, epsilon
        );
        
      
        
        dim3 moeGrid(T, C, C);
        dim3 moeBlock(C, 1, 1);


        SparseMoeBuffers moeBuffers1{
            buffers.moeHiddenX1[l],
            buffers.moeProbs1[l],
            buffers.moeSorted1[l],
            buffers.moeIndices1[l],
            buffers.moeExperts1[l],
            buffers.moeOut1[l]
        };

        SparseMoeParams moeParams1{
            params.moeW1[l],
            params.moeB1[l],
            params.expertW1[l],
            params.expertB1[l]
        };

        SparseMoeBuffers moeBuffers2{
            buffers.moeHiddenX2[l],
            buffers.moeProbs2[l],
            buffers.moeSorted2[l],
            buffers.moeIndices2[l],
            buffers.moeExperts2[l],
            buffers.moeOut2[l]
        };

        SparseMoeParams moeParams2{
            params.moeW2[l],
            params.moeB2[l],
            params.expertW2[l],
            params.expertB2[l]
        };
        
        if(l < L - 1)
        fusedSparsemoeForward<<<moeGrid, moeBlock>>>(
            buffers.moeNormX[l], buffers.attnNormX[l + 1] buffers.moeNorm1[l], params.WNorm[2*l + 1], params.BNorm[2*l + 1],
            moeBuffers1, moeParams1, moeBuffers2, moeParams2, T, C, E, K, epsilon
        );
        else{
            fusedSparsemoeForward<<<moeGrid, moeBlock>>>(
            buffers.moeNormX[l], buffers.decoderOut, buffers.moeNorm1[l], params.WNorm[2*l + 1], params.BNorm[2*l + 1],
            moeBuffers1, moeParams1, moeBuffers2, moeParams2, T, C, E, K, epsilon
        );
        }
            
    }

    // buffers.CLSNormX = buffers.decoderOut + (T - 1) * C;
   

    dim3 gridDim(C, 1, 1);
    dim3 blockDim(C, 1, 1);

    classifierForward<<<gridDim, blockDim>>>(
        buffers.CLSNormX, buffers.CLSNorm, params.WNorm[2*L + 1], params.BNorm[2*L + 1],
        buffers.ZNormCLS, params.WCLS, params.BCLS, buffers.ZCLS, buffers.Y, C, F, epsilon
    );
    


}

void transformerBackward(ModelParameters& params, ModelParameters& gradients, ModelParameters& velocity, ModelParameters& momentum,
     const ModelHyperparameters& config, float learningRate, float beta1, float beta2, float weightDecay, int time,
      IntermediateBuffers& buffers, BackpropagationBuffers& backbuffers){

    const int C = config.C;
    const int T = config.T;
    const int N = config.N;
    const int H = config.H;
    const int E = config.E;
    const int L = config.L;
    const int F = config.F;
    const int K = config.K;
    const float epsilon = config.epsilon;



}



