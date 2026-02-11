#ifndef TRANSFORMER_CU
#define TRANSFORMER_CU

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
#include <cooperative_groups.h>

__device__ int SyncFlag = 0;
__device__ int Count = 0;

#define ceildiv(a, b) ((a + b - 1) / b)


__global__ void calculateTheta(float* theta, int T, int H, int B){
    int token = blockIdx.x;
    int tid = threadIdx.x;
    if(token > T && tid > H / 2) return;
    theta[token * H/2 + tid] = token * powf(B,  -2 * (float)tid / H);
}

__global__ void classifierForward(float* X, float* XNorm, float* WNorm, float* BNorm, float* ZNorm, float* W, float* B, float* Z, float* Y, float* RMS,
     const int C, const int F, const float epsilon){
   

    RMSNormForward<1, 16, 8, 1, 4>(X, XNorm, WNorm, BNorm, ZNorm, RMS, 1, C, epsilon);
    simpleLinear(ZNorm, W, B, Z, 1, F, C); // optimised sgemm is overkill for what is essentially 2 dot products
    await();
    simpleSoftmaxForward(Z, Y);
}

__global__ void classifierBackward(float* Y, float* P, float* dL_dZCLS, float* dL_dW, float* dL_dZNormCLS, float* ZNorm,
     float* W, float* dL_dXNorm, float* XNorm, float* WNorm, float* dL_dWNorm, float* RMS, float* dL_dX, const int C, const int F, const float epsilon){
    

    BCESoftmaxBackward(Y, P, dL_dZCLS, F);
    simpleLinearBackward(dL_dZCLS, dL_dW, dL_dZNormCLS, ZNorm, W, 1, F, C);
    RMSNormBackward<1, 8, 16, 1, 4, 4>(dL_dZNormCLS, dL_dXNorm, dL_dWNorm, dL_dX, XNorm, W, RMS[0], C, epsilon);
}

__global__ void flashAttentionForward(float* X, float* attnNorm, float* WNorm1, float* BNorm1, float* attnX, float* Y, float* RMS1, 
    const AttentionBuffers& buffers, const AttentionParams& params, float* moeNorm, float* moeX, float* WNorm2, float* BNorm2, float* RMS2, 
    const int T, const int C, const int N, const int H, const float epsilon){
  
    
    RMSNormForward<16, 16, 8, 4, 4>(X, attnNorm, WNorm1, BNorm1, attnX, RMS1, T, C, epsilon);
    attentionForward(attnX, buffers, params, T, C, N, H);

    int token = blockIdx.z;
    int tid = threadIdx.x;

    alignas(16) float regX[4];
    alignas(16) float regY[4];
    if(blockIdx.x == 0 && blockIdx.y == 0  && token < T){
        if(tid < (C / 4)){
            reinterpret_cast<float4*>(regX)[0] = reinterpret_cast<float4*>(&X[token * C + tid * 4])[0];
            reinterpret_cast<float4*>(regY)[0] = reinterpret_cast<float4*>(&buffers.projY[token * C + tid * 4])[0];

            #pragma unroll
            for(int i = 0; i < 4; ++i){
                regY[i] += regX[i];
            }
            reinterpret_cast<float4*>(&Y[token * C + tid * 4])[0] = reinterpret_cast<float4*>(regY)[0];
        }
        __syncthreads();
    }
    
    await();
    RMSNormForward<16, 16, 8, 4, 4>(Y, moeNorm, WNorm2, BNorm2, moeX, RMS2, T, C, epsilon);

}


__global__ void flashAttentionBackward(const AttentionBuffers& buffers, const AttentionParams& params, AttentionParams& gradients,
    AttentionBackBuffers& backBuffers, float* dL_dAttnX, float* dL_dXNorm, float* dL_dX, float* XNorm,
     float* AttnX, float* RMS, float* dL_dWNorm, float* WNorm, const int C, const int N, const int H, const int T, const float epsilon){
    
    attentionBackward(buffers, params, gradients, backBuffers, AttnX, dL_dAttnX, C, N, H, T);
    RMSNormBackward<1, 8, 16, 1, 4, 4>(dL_dAttnX, dL_dXNorm, dL_dWNorm, dL_dX, XNorm, WNorm, RMS[0], C, epsilon);
}

__global__ void fusedSparsemoeForward(float* X, float* Y, float* moeHidden,
    const SparseMoeBuffers& buffers1, const SparseMoeParams& params1, const SparseMoeBuffers& buffers2,
    const SparseMoeParams& params2, const int T, const int C, const int E, const int K){

    sparseMoeForward(X, buffers1, params1, T, C, C * 4, E, K);

    int token = blockIdx.x;
    int tid = threadIdx.x;

    alignas(16) float regX[4] = {0.0f};
    alignas(16) float regY[4] = {0.0f};
    if(blockIdx.z == 0 && blockIdx.y == 0 && token < T){
        if(tid < C){ 
            reinterpret_cast<float4*>(regX)[0] = reinterpret_cast<float4*>(&buffers1.Y[token * C * 4 + tid * 4])[0];

            #pragma unroll
            for(int i = 0; i < 4; ++i){
                regY[i] = siluForward(regX[i]);
            }
            reinterpret_cast<float4*>(&moeHidden[token * C * 4 + tid * 4])[0] = reinterpret_cast<float4*>(regY)[0];
            
        }
        __syncthreads();
    }
    await();
    sparseMoeForward(moeHidden, buffers2, params2, T, C * 4, C, E, K);

    if(blockIdx.z == 0 && blockIdx.y == 0 && token < T){
        if(tid < (C / 4)){
            reinterpret_cast<float4*>(regX)[0] = reinterpret_cast<float4*>(&X[token * C + tid * 4])[0];
            reinterpret_cast<float4*>(regY)[0] = reinterpret_cast<float4*>(&buffers2.Y[token * C + tid * 4])[0];

            #pragma unroll
            for(int i = 0; i < 4; ++i){
                regY[i] += regX[i];
            }
            reinterpret_cast<float4*>(&Y[token * C + tid * 4])[0] = reinterpret_cast<float4*>(regY)[0];
            
        }
        __syncthreads();
    }
}

__global__ void fusedSparseMoeBackward(SparseMoeBuffers& moeBuffers2, SparseMoeParams& moeParams2, SparseMoeParams& moeGradients2, SparseMoeBackBuffers& moeBackBuffers2,
     SparseMoeBuffers& moeBuffers1, SparseMoeParams& moeParams1, SparseMoeParams& moeGradients1, SparseMoeBackBuffers& moeBackBuffers1,
     float* dL_dmoeY2, float* dL_dmoeY1, float* dL_dmoeHidden, float* loadBalance2, float* loadBalance1, float* dL_dX,
     float* RMS, float* dL_dWNorm, float* WNorm, float* dL_dXNorm, float* XNorm, float* dL_dBNorm, float* moeHidden, float* moeX,
     const float alpha, const float beta, const int C, const int E, const int K, const int T, const float epsilon){

    loadBalance(loadBalance2, moeBuffers2.indices, T, E, K);
    loadBalance(loadBalance1, moeBuffers1.indices, T, C * 4, K);
    moeBuffers2.indices += (T-1) * E;
    moeBuffers1.indices += (T-1) * E;
    sparseMoeBackward(dL_dmoeY2, dL_dmoeHidden, moeHidden, moeBuffers2, moeParams2, moeGradients2, moeBackBuffers2, loadBalance2, alpha, beta, 4 * C, C, E, K);
    
    alignas(16) float regX[4];
    alignas(16) float regY[4];
    alignas(16) float regGrad[4];
    int tid = threadIdx.x;
    if(tid < C && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0){
        reinterpret_cast<float4*>(&regY)[0] = reinterpret_cast<float4*>(&moeBuffers1.Y[tid * 4])[0];
        reinterpret_cast<float4*>(&regGrad)[0] = reinterpret_cast<float4*>(&dL_dmoeHidden[tid * 4])[0];
        #pragma unroll
        for(int i = 0; i < 4; ++i){
            regX[i] = regGrad[i] * siluBackward(regY[i]);
        }
        reinterpret_cast<float4*>(&dL_dmoeY1[tid * 4])[0] = reinterpret_cast<float4*>(&regX)[0];
    }
    __syncthreads();
    await();
    sparseMoeBackward(dL_dmoeY1, dL_dBNorm, moeX, moeBuffers1, moeParams1, moeGradients1, moeBackBuffers1, loadBalance1, alpha, beta, C, C * 4, E, K);
    RMSNormBackward<1, 8, 16, 1, 4, 4>(dL_dBNorm, dL_dXNorm, dL_dWNorm, dL_dX, XNorm, WNorm, RMS[0], C, epsilon);
}

__global__ void optimizer(ModelParameters& params, ModelParameters& gradients, ModelParameters& velocity, ModelParameters& momentum,
     const ModelHyperparameters& config, float learningRate, float beta1, float beta2, float l2Lambda, int time, float epsilon){
    const int C = config.C;
    const int T = config.T;
    const int N = config.N;
    const int H = config.H;
    const int E = config.E;
    const int L = config.L;
    const int F = config.F;
    const int K = config.K;

    Parameter CLS{
        params.CLS,
        gradients.CLS,
        velocity.CLS,
        momentum.CLS
    };

    optimizerStep(CLS, learningRate, beta1, beta2, epsilon, l2Lambda, time, 1, 1, 1, C);

    Parameter WNorm{
        params.WNorm,
        gradients.WNorm,
        velocity.WNorm,
        momentum.WNorm
    };

    optimizerStep(WNorm, learningRate, beta1, beta2, epsilon, l2Lambda, time, L * 2 + 1, C, 1, C);

    Parameter BNorm{
        params.BNorm,
        gradients.BNorm,
        velocity.BNorm,
        momentum.BNorm
    };

    optimizerStep(BNorm, learningRate, beta1, beta2, epsilon, l2Lambda, time, L * 2 + 1, 1, 1, C);

    Parameter qkvw{
        params.qkvw,
        gradients.qkvw,
        velocity.qkvw,
        momentum.qkvw
    };
    optimizerStep(qkvw, learningRate, beta1, beta2, epsilon, l2Lambda, time, L, C, 1, 3 * C);

    Parameter qkvb{
        params.qkvb,
        gradients.qkvb,
        velocity.qkvb,
        momentum.qkvb
    };

    optimizerStep(qkvb, learningRate, beta1, beta2, epsilon, l2Lambda, time, L, 1, 1, 3 * C);

    Parameter projW{
        params.projW,
        gradients.projW,
        velocity.projW,
        momentum.projW
    };

    optimizerStep(projW, learningRate, beta1, beta2, epsilon, l2Lambda, time, L, C, 1, C);

    Parameter projB{
        params.projB,
        gradients.projB,
        velocity.projB,
        momentum.projB
    };

    optimizerStep(projB, learningRate, beta1, beta2, epsilon, l2Lambda, time, L, 1, 1, C);

    Parameter moegw1{
        params.moegw1,
        gradients.moegw1,
        velocity.moegw1,
        momentum.moegw1
    };

    optimizerStep(moegw1, learningRate, beta1, beta2, epsilon, l2Lambda, time, L, C, 1, E);

    Parameter moegb1{
        params.moegb1,
        gradients.moegb1,
        velocity.moegb1,
        momentum.moegb1
    };

    optimizerStep(moegb1, learningRate, beta1, beta2, epsilon, l2Lambda, time, L, 1, 1, E);

    Parameter moew1{
        params.moew1,
        gradients.moew1,
        velocity.moew1,
        momentum.moew1
    };

    optimizerStep(moew1, learningRate, beta1, beta2, epsilon, l2Lambda, time, L, E, C, C * 4);

    Parameter moeb1{
        params.moeb1,
        gradients.moeb1,
        velocity.moeb1,
        momentum.moeb1
    };

    optimizerStep(moeb1, learningRate, beta1, beta2, epsilon, l2Lambda, time, L, E, 1, C * 4);

    Parameter moegw2{
        params.moegw2,
        gradients.moegw2,
        velocity.moegw2,
        momentum.moegw2
    };

    optimizerStep(moegw2, learningRate, beta1, beta2, epsilon, l2Lambda, time, L, C * 4, 1, E);

    Parameter moegb2{
        params.moegb2,
        gradients.moegb2,
        velocity.moegb2,
        momentum.moegb2
    };

    optimizerStep(moegb2, learningRate, beta1, beta2, epsilon, l2Lambda, time, L, 1, 1, E);

    Parameter moew2{
        params.moew2,
        gradients.moew2,
        velocity.moew2,
        momentum.moew2
    };

    optimizerStep(moew2, learningRate, beta1, beta2, epsilon, l2Lambda, time, L, E, C * 4, C);

    Parameter moeb2{
        params.moeb2,
        gradients.moeb2,
        velocity.moeb2,
        momentum.moeb2
    };

    optimizerStep(moeb2, learningRate, beta1, beta2, epsilon, l2Lambda, time, L, E, 1, C);

    Parameter WCLS{
        params.WCLS,
        gradients.WCLS,
        velocity.WCLS,
        momentum.WCLS
    };

    optimizerStep(WCLS, learningRate, beta1, beta2, epsilon, l2Lambda, time, 1, 1, C, F);

    Parameter BCLS{
        params.BCLS,
        gradients.BCLS,
        velocity.BCLS,
        momentum.BCLS
    };

    optimizerStep(BCLS, learningRate, beta1, beta2, epsilon, l2Lambda, time, 1, 1, 1, F);

}   


void calculateThetaLauncher(float* theta, int T, int H, int B){
    dim3 gridDim(T, 1, 1);
    dim3 blockDim(H / 2, 1, 1);

    calculateTheta<<<gridDim, blockDim>>>(theta, T, H, B);
}

void calculateLoss(float* Y, float* P, float* loss, int F, float epsilon){
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(F, 1, 1);

    binaryCrossEntropyLossForward<<<gridDim, blockDim>>>(Y, P, loss, F, epsilon);
}
void transformerForward(float* X, float* Y, ModelParameters& params, const ModelHyperparameters& config, IntermediateBuffers& buffers){

    const int C = config.C;
    const int T = config.T;
    const int N = config.N;
    const int H = config.H;
    const int E = config.E;
    const int L = config.L;
    const int F = config.F;
    const int K = config.K;
    const float epsilon = config.epsilon;
    

    
    dim3 attnGrid(N, C/16, T);
    dim3 attnBlock(C, 1, 1);  

    AttentionBuffers* attnBuffers;
    cudaMallocManaged(&attnBuffers, sizeof(AttentionBuffers));

    AttentionParams* attnParams;
    cudaMallocManaged(&attnParams, sizeof(AttentionParams));

    dim3 moeGrid(T, C/16, C/16);
    dim3 moeBlock(C*4, 1, 1);

    SparseMoeBuffers* moeBuffers1;
    cudaMallocManaged(&moeBuffers1, sizeof(SparseMoeBuffers));

    SparseMoeParams* moeParams1;
    cudaMallocManaged(&moeParams1, sizeof(SparseMoeParams));

    SparseMoeBuffers* moeBuffers2;
    cudaMallocManaged(&moeBuffers2, sizeof(SparseMoeBuffers));

    SparseMoeParams* moeParams2;
    cudaMallocManaged(&moeParams2, sizeof(SparseMoeParams));

    for(int l = 0; l < L; ++l){
        
        *attnBuffers = {
            buffers.theta,
            buffers.qkv + l * N * T * H * 3,
            buffers.Q + l * N * T * H,
            buffers.K + l * N * T * H,
            buffers.V + l * N * T * H,
            buffers.QR + l * N * T * H,
            buffers.KR + l * N * T * H,
            buffers.KT + l * N * T * H,
            buffers.attnScores + l * N * T * T,
            buffers.attnWeights + l * N * T * T,
            buffers.attnY + l * N * T * H,
            buffers.projX+ l * T * C,
            buffers.projY + l * T * C
        };

        *attnParams = {
            params.qkvw + l * C * 3 * C,
            params.qkvb + l * 3 * C,
            params.projW + l * C * C,
            params.projB + l * C
        };
       
        if(l == 0){
            flashAttentionForward<<<attnGrid, attnBlock>>>(
            X, buffers.attnNorm + l * T * C, params.WNorm + 2*l * C * C, params.BNorm + 2*l * C,
            buffers.attnX + l * T * C, buffers.moeNormX + l * T * C, buffers.RMS + 2 * l, *attnBuffers, *attnParams,
            buffers.moeNorm + l * T * C, buffers.moeX + l * T * C, params.WNorm + (2*l + 1) * C * C, params.BNorm + (2*l + 1) * C, buffers.RMS + (2*l + 1), 
            T, C, N, H, epsilon
            );
        }
        else{
         
            flashAttentionForward<<<attnGrid, attnBlock>>>(
                buffers.attnNormX + (l - 1) * T * C, buffers.attnNorm + l * T * C, params.WNorm + 2*l * C * C, params.BNorm + 2*l * C,
                buffers.attnX + l * T * C, buffers.moeNormX + l * T * C, buffers.RMS + 2 * l, *attnBuffers, *attnParams, 
                buffers.moeNorm + l * T * C, buffers.moeX + l * T * C, params.WNorm + (2*l + 1) * C * C, params.BNorm + (2*l + 1) * C, buffers.RMS + (2*l + 1),
                 T, C, N, H, epsilon
            );
        }

        cudaDeviceSynchronize();


        *moeBuffers1 = {
            buffers.moeGate1 + l * T * E,
            buffers.moeProbs1 + l * T * E,
            buffers.moeSorted1 + l * T * E,
            buffers.moeIndices1 + l * T * E,
            buffers.moeW1 + l * K * T * C * C * 4,
            buffers.moeB1 + l * K * T * C * 4,
            buffers.moeExperts1 + l * T * K * C * 4,
            buffers.moeY1 + l * T * C * 4
        };

        *moeParams1 = {
            params.moegw1 + l * C * E,
            params.moegb1 + l * E,
            params.moew1 + l * E * C * C * 4,
            params.moeb1 + l * E * C * 4
        };

        *moeBuffers2 = {
            buffers.moeGate2 + l * T * E,
            buffers.moeProbs2 + l * T * E,
            buffers.moeSorted2 + l * T * E,
            buffers.moeIndices2 + l * T * E,
            buffers.moeW2 + l * K * T * C * C * 4,
            buffers.moeB2 + l * K * T * C,
            buffers.moeExperts2 + l * T * K * C,
            buffers.moeY2 + l * T * C 
        };

        *moeParams2 = {
            params.moegw2 + l * C * 4 * E,
            params.moegb2 + l * E,
            params.moew2 + l * E * C * 4 * C,
            params.moeb2 + l * E * C
        };
        
       
        if(l < L - 1)
            fusedSparsemoeForward<<<moeGrid, moeBlock>>>(
            buffers.moeX + l * T * C, buffers.attnNormX + l * T * C,   buffers.moeHidden + l * T * C * 4, *moeBuffers1, *moeParams1, *moeBuffers2, *moeParams2, T, C, E, K
        );
        else{
            fusedSparsemoeForward<<<moeGrid, moeBlock>>>(
            buffers.moeX + l * T * C, buffers.decoderY,   buffers.moeHidden + l * T * C * 4,  *moeBuffers1, *moeParams1, *moeBuffers2, *moeParams2, T, C, E, K
            );
        }
        cudaDeviceSynchronize();

    }

    cudaFree(attnBuffers);
    cudaFree(attnParams);
    cudaFree(moeBuffers1);
    cudaFree(moeParams1);
    cudaFree(moeBuffers2);
    cudaFree(moeParams2);

    dim3 gridDim(C, 1, 1);
    dim3 blockDim(C, 1, 1);
    classifierForward<<<gridDim, blockDim>>>(
        buffers.decoderY + (T - 1) * C, buffers.CLSNorm, params.WNorm + 2*L * C * C, params.BNorm + 2*L * C,
        buffers.ZNormCLS, params.WCLS, params.BCLS, buffers.ZCLS, Y, buffers.RMS + 2*L, C, F, epsilon
    );
    
    
}

void transformerBackward(ModelParameters& params, ModelParameters& gradients, ModelParameters& velocity, ModelParameters& momentum,
     const ModelHyperparameters& config, float learningRate, float alpha, float beta, float beta1, float beta2, float l2Lambda, int time,
      IntermediateBuffers& buffers, BackpropagationBuffers& backbuffers, float* Y, float* P){

    const int C = config.C;
    const int T = config.T;
    const int N = config.N;
    const int H = config.H;
    const int E = config.E;
    const int L = config.L;
    const int F = config.F;
    const int K = config.K;
    const float epsilon = config.epsilon;
    



    dim3 gridDim(1, C/16, C/16);
    dim3 blockDim(C, C, 1);
    classifierBackward<<<gridDim, blockDim>>>(
        Y, P, gradients.BCLS, gradients.WCLS, gradients.BNorm + 2*L * C, buffers.ZNormCLS,
        params.WCLS, backbuffers.dL_dXNorm + 2*L * C, buffers.decoderY + (T - 1) * C, params.WNorm + 2*L * C * C,
        gradients.WNorm + 2*L * C * C, buffers.RMS + 2*L, backbuffers.dL_dMoeY2 + (L - 1) * C,  C, F, epsilon
    );
    printf("Finished classifier backward\n");

    cudaDeviceSynchronize();

    printf("Device synchronized after classifier backward\n ");
    dim3 moeGrid(E, C/16, C/16);
    dim3 moeBlock(C, C, 1);

    SparseMoeBuffers* moeBuffers1;
    cudaMallocManaged(&moeBuffers1, sizeof(SparseMoeBuffers));

    SparseMoeParams* moeParams1;
    cudaMallocManaged(&moeParams1, sizeof(SparseMoeParams));

    SparseMoeParams* moeGradients1;
    cudaMallocManaged(&moeGradients1, sizeof(SparseMoeParams));

    SparseMoeBackBuffers* moeBackBuffers1;
    cudaMallocManaged(&moeBackBuffers1, sizeof(SparseMoeBackBuffers));

    SparseMoeBuffers* moeBuffers2;
    cudaMallocManaged(&moeBuffers2, sizeof(SparseMoeBuffers));

    SparseMoeParams* moeParams2;
    cudaMallocManaged(&moeParams2, sizeof(SparseMoeParams));

    SparseMoeParams* moeGradients2;
    cudaMallocManaged(&moeGradients2, sizeof(SparseMoeParams));

    SparseMoeBackBuffers* moeBackBuffers2;
    cudaMallocManaged(&moeBackBuffers2, sizeof(SparseMoeBackBuffers));

    dim3 attnGrid(N, C/16, T);
    dim3 attnBlock(C, C, 1); 

    AttentionBuffers* attnBuffers;
    cudaMallocManaged(&attnBuffers, sizeof(AttentionBuffers));

    AttentionParams* attnParams;
    cudaMallocManaged(&attnParams, sizeof(AttentionParams));

    AttentionParams* attnGradients;
    cudaMallocManaged(&attnGradients, sizeof(AttentionParams));

    AttentionBackBuffers* attnBackBuffers;
    cudaMallocManaged(&attnBackBuffers, sizeof(AttentionBackBuffers));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {   
        printf("CUDA error after malloc: %s\n", cudaGetErrorString(err));
    }
    for(int l = L - 1; l >= 0; --l){


        *moeBuffers1 = {
            buffers.moeGate1 + l * T * E + (T - 1) * E,
            buffers.moeProbs1 + l * T * E + (T - 1) * E,
            buffers.moeSorted1 + l * T * E + (T - 1) * E,
            buffers.moeIndices1 + l * T * E,
            buffers.moeW1 + l * K * T * C * C * 4 + (T - 1) * K * C * C * 4,
            buffers.moeB1 + l * K * T * C * 4 + (T - 1) * K * C * 4,
            buffers.moeExperts1 + l * T * K * C * 4 + (T - 1) * K * C * 4,
            buffers.moeY1 + l * T * C * 4 + (T - 1) * C * 4
        };

        *moeParams1 = {
            params.moegw1 + l * C * E,
            params.moegb1 + l * E,
            params.moew1 + l * E * C * C * 4,
            params.moeb1 + l * E * C * 4
        };

        *moeGradients1 = {
            gradients.moegw1 + l * C * E,
            gradients.moegb1 + l * E,
            gradients.moew1 + l * E * C * C * 4,
            gradients.moeb1 + l * E * C * 4
        };

        *moeBackBuffers1 = {
            backbuffers.Xtmp1 + l * K * C,
            backbuffers.dL_dWtmp1 + l * K * C * C * 4,
            backbuffers.dL_dBtmp1 + l * K * C * 4,
            backbuffers.dE_dX1 + l * K * C,
            backbuffers.Wtmp1 + l * K * C * C * 4,
            backbuffers.Etmp1 + l * K * C * 4,
            backbuffers.EMA1 + l * E * C * 4, 
            backbuffers.J1 + l,
            backbuffers.dL_dG1 + l * E,
            backbuffers.dG_dX1 + l * K * C,
            backbuffers.dG_dZ1 + l * E,
            backbuffers.dL_dXtmp1 + l * K * C,
            backbuffers.dL_dXsum1 + l * C
        };

        *moeBuffers2 = {
            buffers.moeGate2 + l * T * E + (T - 1) * E,
            buffers.moeProbs2 + l * T * E + (T - 1) * E,
            buffers.moeSorted2 + l * T * E + (T - 1) * E,
            buffers.moeIndices2 + l * T * E,
            buffers.moeW2 + l * K * T * C * C * 4 + (T - 1) * K * C * C * 4,
            buffers.moeB2 + l * K * T * C + (T - 1) * K * C,
            buffers.moeExperts2 + l * T * K * C + (T - 1) * K * C,
            buffers.moeY2 + l * T * C + (T - 1) * C 
        };

        *moeParams2 = {
            params.moegw2 + l * C * 4 * E,
            params.moegb2 + l * E,
            params.moew2 + l * E * C * 4 * C,
            params.moeb2 + l * E * C
        };

        *moeGradients2 = {
            gradients.moegw2 + l * C * 4 * E,
            gradients.moegb2 + l * E,
            gradients.moew2 + l * E * C * 4 * C,
            gradients.moeb2 + l * E * C
        };

        *moeBackBuffers2 = {
            backbuffers.Xtmp2 + l * K * C * 4,
            backbuffers.dL_dWtmp2 + l * K * C * 4 * C,
            backbuffers.dL_dBtmp2 + l * K * C,
            backbuffers.dE_dX2 + l * K * C * 4,
            backbuffers.Wtmp2 + l * K * C * 4 * C,
            backbuffers.Etmp2 + l * K * C,
            backbuffers.EMA2 + l * E * C, 
            backbuffers.J2 + l,
            backbuffers.dL_dG2 + l * E,
            backbuffers.dG_dX2 + l * K * C * 4,
            backbuffers.dG_dZ2 + l * E,
            backbuffers.dL_dXtmp2 + l * K * C * 4,
            backbuffers.dL_dXsum2 + l * C * 4
        };
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error after setting up MoE buffers at layer %d: %s\n", l, cudaGetErrorString(err));
        }
        printf("layer %d\n", l);

        fusedSparseMoeBackward<<<moeGrid, moeBlock>>>(
            *moeBuffers2, *moeParams2, *moeGradients2, *moeBackBuffers2, *moeBuffers1, *moeParams1, *moeGradients1, *moeBackBuffers1,
            backbuffers.dL_dMoeY2 + l * C, backbuffers.dL_dMoeY1 + l * C * 4, backbuffers.dL_dMoeHidden + l * C * 4,
            backbuffers.loadBalance2 + l * E, backbuffers.loadBalance1 + l * E, gradients.projB + l * C,
            buffers.RMS + 2*l + 1, gradients.WNorm + (2* l + 1) * C * C, params.WNorm + (2* l + 1) * C * C,
             backbuffers.dL_dXNorm + (2 * l + 1) * C, buffers.moeNorm + l * C, gradients.BNorm + (2*l + 1) * C,
              buffers.moeHidden + l * C * 4, buffers.moeX + l * C, alpha, beta, C, E, K, T, epsilon
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in fusedSparseMoeBackward at layer %d: %s\n", l, cudaGetErrorString(err));
        }
        printf("Finished MoE layer %d backward\n", l);
        cudaDeviceSynchronize();
        printf("Device sync\n");    
        *attnBuffers = {
            buffers.qkv + l * N * T * H * 3,
            buffers.Q + l * N * T * H,
            buffers.K + l * N * T * H,
            buffers.V + l * N * T * H,
            buffers.attnScores + l * N * T * T,
            buffers.attnWeights + l * N * T * H,
            buffers.attnY + l * N * T * H,
            buffers.projX+ l * T * C,
            buffers.projY + l * T * C
        };

        *attnParams = {
            params.qkvw + l * C * 3 * C,
            params.qkvb + l * 3 * C,
            params.projW + l * C * C,
            params.projB + l * C
        };

        *attnGradients = {
            gradients.qkvw + l * C * 3 * C,
            gradients.qkvb + l * 3 * C,
            gradients.projW + l * C * C,
            gradients.projB + l * C
        };

        *attnBackBuffers = {
            backbuffers.dL_dProjX + l * T * C,
            backbuffers.dL_dY + l * N * T * T,
            backbuffers.dL_dW + l * N * T * T,
            backbuffers.dL_dV + l * N * T * H,
            backbuffers.dL_dK + l * N * T * H,
            backbuffers.dL_dQ + l * N * T * H,
            backbuffers.J + l * N * T,
            backbuffers.dL_dScores + l * N * T * T,
            backbuffers.dL_dKT + l * N * T * H,
            backbuffers.dL_dKR + l * N * T * H,
            backbuffers.dL_dQR + l * N * T * H,
            backbuffers.dL_dqkv + l * N * H * 3,
            backbuffers.dL_dXtmp + l * 3 * C
        };

        if(l == 0){
            flashAttentionBackward<<<attnGrid, attnBlock>>>(
                *attnBuffers, *attnParams, *attnGradients, *attnBackBuffers,
                gradients.BNorm + 2*l * C, backbuffers.dL_dXNorm + l * C, gradients.CLS, buffers.attnNorm + l * C,
                buffers.attnX + l * C, buffers.RMS + 2*l, gradients.WNorm + 2*l * C * C, params.WNorm + 2*l * C * C, C, N, H, T, epsilon
            );
        }
        else{
            flashAttentionBackward<<<attnGrid, attnBlock>>>(
                *attnBuffers, *attnParams, *attnGradients, *attnBackBuffers,
                gradients.BNorm + 2*l * C, backbuffers.dL_dXNorm + l * C, backbuffers.dL_dMoeY2 + (l - 1) * C, buffers.attnNorm + l * C,
                buffers.attnX + l * C, buffers.RMS + 2*l, gradients.WNorm + 2*l * C * C, params.WNorm + 2*l * C * C, C, N, H, T, epsilon
            );
        }
        printf("Finished attention layer %d backward\n", l);
        cudaDeviceSynchronize();
        printf("Device sync\n");

    }
    cudaFree(attnBuffers);
    cudaFree(attnParams);
    cudaFree(attnGradients);
    cudaFree(attnBackBuffers);
    cudaFree(moeBuffers1);
    cudaFree(moeParams1);
    cudaFree(moeGradients1);
    cudaFree(moeBackBuffers1);
    cudaFree(moeBuffers2);
    cudaFree(moeParams2);
    cudaFree(moeGradients2);
    cudaFree(moeBackBuffers2);

    dim3 optimGrid(2 * L + 1, 4*C, 4*C);
    dim3 optimBlock(4*C, 1, 1);
    optimizer<<<optimGrid, optimBlock>>>(params, gradients, velocity, momentum, config, learningRate, beta1, beta2, l2Lambda, time, epsilon);
    printf("Finished optimizer step\n");
}



#endif