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

#define ceildiv(a, b) ((a + b - 1) / b)


__global__ void calculateTheta(float* theta, int T, int H, int B){
    int token = blockIdx.x;
    int tid = threadIdx.x;
    if(token > T && tid > H / 2) return;
    theta[token * H/2 + tid] = token * powf(B,  2 * (float)tid / H);
}

__global__ void classifierForward(float* X, float* XNorm, float* WNorm, float* BNorm, float* ZNorm, float* W, float* B, float* Z, float* Y, float* RMS, int C, int F, float epsilon){
    RMSNormForward<1, 32, 8, 1, 4>(X, XNorm, WNorm, BNorm, ZNorm, RMS, 1, C, epsilon);
    simpleLinear(ZNorm, W, B, Z, 1, F, C); // optimised sgemm is overkill for what is essentially 2 dot products
    simpleSoftmaxForward(Z, Y);
}

__global__ void classifierBackward(float* Y, float* P, float* dL_dZCLS, float* dL_dW, float* dL_dZNormCLS, float* ZNorm,
     float* W, float* dL_dXNorm, float* XNorm, float* WNorm, float* dL_dWNorm, float* RMS, float* dL_dX, int C, int F, float epsilon){
    BCESoftmaxBackward(Y, P, dL_dZCLS, F);
    simpleLinearBackward(dL_dZCLS, dL_dW, dL_dZNormCLS, ZNorm, W, 1, F, C);
    RMSNormBackward<1, 32, 1, 1, 4>(dL_dZNormCLS, dL_dXNorm, dL_dWNorm, dL_dX, XNorm, RMS, C, epsilon);
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


__global__ void flashAttentionBackward(const AttentionBuffers& buffers, const AttentionParams& params, AttentionParams& gradients,
    AttentionBackBuffers& backBuffers, float* dL_dAttnX, float* dL_dXNorm, float* dL_dX, float* XNorm,
     float* AttnX, float* RMS, float* dL_dWNorm, float* WNorm, int C, int N, int H, int T, float epsilon){
    attentionBackward(buffers, params, gradients, backBuffers, AttnX, dL_dAttnX, C, N, H, T);
    RMSNormBackward<1, 16, 1, 1, 4>(dL_dAttnX, dL_dXNorm, dL_dWNorm, dL_dX, XNorm, WNorm, RMS, C, epsilon);
}

__global__ void fusedSparsemoeForward(float* X, float* Y, float* moeNorm, float* moeX, float* moeHidden, float* WNorm, float* BNorm, 
    const SparseMoeBuffers& buffers1, const SparseMoeParams& params1, const SparseMoeBuffers& buffers2,
    const SparseMoeParams& params2, int T, int C, int E, int K, float epsilon){
    RMSNormForward<16, 16, 8, 4, 4>(X, moeNorm, WNorm, BNorm, moeX, 1, C, epsilon);
    sparseMoeForward(moeX, buffers1, params1, T, C, C * 4, E, K);
    int row = blockIdx.z;
    int tid = threadIdx.x;

    float regX[4];
    float regY[4] = {0.0f};
    if(tid < C && row < T){
        reinterpret_cast<float4*>(regX)[0] = reinterpret_cast<float4*>(&X[row * C * 4 + tid * 4])[0];
        __syncthreads();

        #pragma unroll
        for(int i = 0; i < 4; ++i){
            regY[i] = siluForward(regX[i]);
        }
        reinterpret_cast<float4*>(&moeHidden[row * C * 4 + tid * 4])[0] = regY;
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

__global__ void fusedSparseMoeBackward(SparseMoeBuffers& moeBuffers2, SparseMoeParams& moeParams2, SparseMoeParams& moeGradients2, SparseMoeBackBuffers& moeBackBuffers2,
     SparseMoeBuffers& moeBuffers1, SparseMoeParams& moeParams1, SparseMoeParams& moeGradients1, SparseMoeBackBuffers& moeBackBuffers1,
     float* dL_dmoeY2, float* dL_dmoeY1, float* dL_dmoeHidden, float* loadBalance2, float* loadBalance1, float* dL_dX,
     float* RMS, float* dL_dWNorm, float* WNorm, float* dL_dXNorm, float* XNorm, float* dL_dBNorm, int C, int E, int K, int T, float epsilon){
    
    loadBalance(loadBalance2, moeBuffers2.indices, T, E, K);
    loadBalance(loadBalance1, moeBuffers1.indices, T, C * 4, K);
    moeBuffers2.indices += (T-1) * E;
    moeBuffers1.indices += (T-1) * E;
    SparseMoeBackward(dL_dmoeY2, dL_dmoeHidden, moeBuffers2, moeParams2, moeGradients2, moeBackBuffers2, 4 * C, C, E, K);
    
    float regX[4];
    float regY[4];
    float regGrad[4];
    int tid = threadIdx.x;
    if(tid < C){
        reinterpret_cast<float4*>(&regY)[0] = reinterpret_cast<float4*>(&moeBuffers1.Y[tid * 4])[0];
        reinterpret_cast<float4*>(&regGrad)[0] = reinterpret_cast<float4*>(&dL_dmoeHidden[tid * 4])[0];
        #pragma unroll
        for(int i = 0; i < 4; ++i){
            regX[i] = regGrad[i] * siluBackward(regY[i]);
        }
        reinterpret_cast<float4*>(&dL_dmoeY1[tid * 4])[0] = reinterpret_cast<float4*>(&regX)[0];
    }

    SparseMoeBackward(dL_dmoeY1, dL_dBNorm, moeBuffers1, moeParams1, moeGradients1, moeBackBuffers1, C, C * 4, E, K);
    RMSNormBackward<1, 16, 1, 1, 4>(dL_dBNorm, dL_dXNorm, dL_dWNorm, dL_dX, XNorm, WNorm, RMS, C, epsilon);
}

__global__ void optimizer(ModelParameters& params, ModelParameters& gradients, ModelParameters& velocity, ModelParameters& momentum,
     const ModelHyperparameters& config, float learningRate, float beta1, float beta2, float weightDecay, int time){
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
    }

    optimizerStep(CLS, learningRate, beta1, beta2, l2Lambda, time, 1, 1, 1, C);

    Parameter WNorm{
        params.WNorm,
        gradients.WNorm,
        velocity.WNorm,
        momentum.WNorm
    }

    optimizerStep(WNorm, learningRate, beta1, beta2, l2Lambda, time, L * 2 + 1, C, 1, C);

    Parameter BNorm{
        params.BNorm,
        gradients.BNorm,
        velocity.BNorm,
        momentum.BNorm
    }

    optimizerStep(BNorm, learningRate, beta1, beta2, l2Lambda, time, L * 2 + 1, 1, 1, C);

    Parameter qkvw{
        params.qkvw,
        gradients.qkvw,
        velocity.qkvw,
        momentum.qkvw
    }
    optimizerStep(qkvw, learningRate, beta1, beta2, l2Lambda, time, L, C, 1, 3 * C);

    Parameter qkvb{
        params.qkvb,
        gradients.qkvb,
        velocity.qkvb,
        momentum.qkvb
    }

    optimizerStep(qkvb, learningRate, beta1, beta2, l2Lambda, time, L, 1, 1, 3 * C);

    Parameter projW{
        params.projW,
        gradients.projW,
        velocity.projW,
        momentum.projW
    }

    optimizerStep(projW, learningRate, beta1, beta2, l2Lambda, time, L, C, 1, C);

    Parameter projB{
        params.projB,
        gradients.projB,
        velocity.projB,
        momentum.projB
    }

    optimizerStep(projB, learningRate, beta1, beta2, l2Lambda, time, L, 1, 1, C);

    Parameter moegw1{
        params.moegw1,
        gradients.moegw1,
        velocity.moegw1,
        momentum.moegw1
    }

    optimizerStep(moegw1, learningRate, beta1, beta2, l2Lambda, time, L, C, 1, E);

    Parameter moegb1{
        params.moegb1,
        gradients.moegb1,
        velocity.moegb1,
        momentum.moegb1
    }

    optimizerStep(moegb1, learningRate, beta1, beta2, l2Lambda, time, L, 1, 1, E);

    Parameter moew1{
        params.moew1,
        gradients.moew1,
        velocity.moew1,
        momentum.moew1
    }

    optimizerStep(moew1, learningRate, beta1, beta2, l2Lambda, time, L, E, C, C * 4);

    Parameter moeb1{
        params.moeb1,
        gradients.moeb1,
        velocity.moeb1,
        momentum.moeb1
    }

    optimizerStep(moeb1, learningRate, beta1, beta2, l2Lambda, time, L, E, 1, C * 4);

    Parameter moegw2{
        params.moegw2,
        gradients.moegw2,
        velocity.moegw2,
        momentum.moegw2
    }

    optimizerStep(moegw2, learningRate, beta1, beta2, l2Lambda, time, L, C * 4, 1, E);

    Parameter moegb2{
        params.moegb2,
        gradients.moegb2,
        velocity.moegb2,
        momentum.moegb2
    }

    optimizerStep(moegb2, learningRate, beta1, beta2, l2Lambda, time, L, 1, 1, E);

    Parameter moew2{
        params.moew2,
        gradients.moew2,
        velocity.moew2,
        momentum.moew2
    }

    optimizerStep(moew2, learningRate, beta1, beta2, l2Lambda, time, L, E, C * 4, C);

    Parameter moeb2{
        params.moeb2,
        gradients.moeb2,
        velocity.moeb2,
        momentum.moeb2
    }

    optimizerStep(moeb2, learningRate, beta1, beta2, l2Lambda, time, L, E, 1, C);

    Parameter WCLS{
        params.WCLS,
        gradients.WCLS,
        velocity.WCLS,
        momentum.WCLS
    }

    optimizerStep(WCLS, learningRate, beta1, beta2, l2Lambda, time, C, 1, 1, F);

    Parameter BCLS{
        params.BCLS,
        gradients.BCLS,
        velocity.BCLS,
        momentum.BCLS
    }

    optimizerStep(BCLS, learningRate, beta1, beta2, l2Lambda, time, 1, 1, 1, F);

}   


void calculateThetaLauncher(float* theta, int T, int H, int B){
    dim3 gridDim(T, 1, 1);
    dim3 blockDim(H / 2, 1, 1);

    calculateTheta<<<gridDim, blockDim>>>(theta, T, H, B);
}

void calculateLoss(float* Y, float* P, float* loss, int F){
    dim3 gridDim(1, 1, 1);
    dim3 blockDim(F, 1, 1);

    BCESoftmaxLoss<<<gridDim, blockDim>>>(Y, P, loss, F);
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

    

    dim3 attnGrid(N, C, T);
    dim3 attnBlock(C, 1, 1);  



    dim3 moeGrid(T, C, C);
    dim3 moeBlock(C, 1, 1);

    for(int l = 0; l < L; ++l){
       

        AttentionBuffers attnBuffers{
            buffers.qkv + l * N * T * H * 3,
            buffers.Q + l * N * T * H,
            buffers.K + l * N * T * H,
            buffers.V + l * N * T * H,
            buffers.attnScores + l * N * T * T,
            buffers.W + l * N * T * H,
            buffers.attnY + l * N * T * H,
            buffers.projX+ l * T * C,
            buffers.projY + l * T * C
        };

        AttentionParams attnParams{
            params.qkvw + l * C * 3 * C,
            params.qkvb + l * 3 * C,
            params.projW + l * C * C,
            params.projB + l * C
        };

        if(l == 0){
            flashAttentionForward<<<attnGrid, attnBlock>>>(
            X, buffers.attnNorm + l * T * C, params.WNorm + 2*l * C * C, params.BNorm + 2*l * C,
            buffers.attnX + l * T * C, buffers.moeHiddenX1 + l * T * C, attnBuffers, attnParams, T, C, N, H, epsilon
        );}
        else{
         
            flashAttentionForward<<<attnGrid, attnBlock>>>(
                buffers.attnNormX + (l - 1) * T * C, buffers.attnNorm + l * T * C, params.WNorm + 2*l * C * C, params.BNorm + 2*l * C,
                buffers.attnX + l * T * C, buffers.moeHiddenX1 + l * T * C, attnBuffers, attnParams, T, C, N, H, epsilon
            );
        }
        
      
        

        SparseMoeBuffers moeBuffers1{
            buffers.moeGate1 + l * T * E,
            buffers.moeProbs1 + l * T * E,
            buffers.moeSorted1 + l * T * E,
            buffers.moeIndices1 + l * T * E,
            buffers.moeW1 + l * K * T * C * C * 4,
            buffers.moeB1 + l * K * T * C * 4,
            buffers.moeExperts1 + l * T * K * C * 4,
            buffers.moeY1 + l * T * C * 4
        };

        SparseMoeParams moeParams1{
            params.moegw1 + l * C * E,
            params.moeb1 + l * E,
            params.moew1 + l * E * C * C * 4,
            params.moeb1 + l * E * C * 4
        };

        SparseMoeBuffers moeBuffers2{
            buffers.moeGate2 + l * T * E,
            buffers.moeProbs2 + l * T * E,
            buffers.moeSorted2 + l * T * E,
            buffers.moeIndices2 + l * T * E,
            buffers.moeW2 + l * K * T * C * C * 4,
            buffers.moeB2 + l * K * T * C,
            buffers.moeExperts2 + l * T * K * C,
            buffers.moeY2 + l * T * C 
        };

        SparseMoeParams moeParams2{
            params.moegw2 + l * C * 4 * E,
            params.moeb2 + l * E,
            params.moew2 + l * E * C * 4 * C,
            params.moeb2 + l * E * C
        };
        
        if(l < L - 1)
        fusedSparsemoeForward<<<moeGrid, moeBlock>>>(
            buffers.moeNormX + l * T * C, buffers.attnNormX + l * T * C, buffers.moeNorm1 + l * T * C, buffers.moeX + l * T * C, buffers.moeHidden + l * T * C * 4, params.WNorm + (2*l + 1) * C * C, params.BNorm + (2*l + 1) * C,
             moeBuffers1, moeParams1, moeBuffers2, moeParams2, T, C, E, K, epsilon
        );
        else{
            fusedSparsemoeForward<<<moeGrid, moeBlock>>>(
            buffers.moeNormX + l * T * C, buffers.decoderY, buffers.moeNorm1 + l * T * C, buffers.moeX + l * T * C, buffers.moeHidden + l * T * C * 4, params.WNorm + (2*l + 1) * C * C, params.BNorm + (2*l + 1) * C,
            moeBuffers1, moeParams1, moeBuffers2, moeParams2, T, C, E, K, epsilon
            );
        }
            
    }
   

    dim3 gridDim(C, 1, 1);
    dim3 blockDim(C, 1, 1);

    classifierForward<<<gridDim, blockDim>>>(
        buffers.decoderY + (T - 1) * C, buffers.CLSNorm, params.WNorm + 2*L * C * C, params.BNorm + 2*L * C,
        buffers.ZNormCLS, params.WCLS, params.BCLS, buffers.ZCLS, Y, buffers.RMS + 2*L, C, F, epsilon
    );
    


}

void transformerBackward(ModelParameters& params, ModelParameters& gradients, ModelParameters& velocity, ModelParameters& momentum,
     const ModelHyperparameters& config, float learningRate, float beta1, float beta2, float l2Lambda, int time,
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
    



    dim3 gridDim(1, C, C);
    dim3 blockDim(C, 1, 1);
    classifierBackward<<<gridDim, blockDim>>>(
        Y, P, gradients.BCLS, gradients.WCLS, gradients.BNorm + 2*L * C, buffers.ZNormCLS,
        params.WCLS, backbuffers.dL_dXNormCLS, buffers.decoderY + (T - 1) * C, params.WNorm + 2*L * C * C,
        gradients.WNorm + 2*L * C * C, buffers.RMS + 2*L, backbuffers.dL_moeY[L - 1], C, F, epsilon
    );


    dim3 moeGrid(E, 4*C, 4*C);
    dim3 moeBlock(C, 1, 1);


    dim3 attnGrid(N, C, T);
    dim3 attnBlock(C, 1, 1); 



    for(int l = L - 1; l >= 0; --l){


        SparseMoeBuffers moeBuffers1{
            buffers.moeGate1 + l * (T - 1) * E,
            buffers.moeProbs1 + l * (T - 1) * E,
            buffers.moeSorted1 + l * (T - 1) * E,
            buffers.moeIndices1 + l * (T - 1) * E,
            buffers.moeW1 + l * K * (T - 1) * C * C * 4,
            buffers.moeB1 + l * K * (T - 1) * C * 4,
            buffers.moeExperts1 + l * (T - 1) * K * C * 4,
            buffers.moeY1 + l * (T - 1) * C * 4
        };

        SparseMoeParams moeParams1{
            params.moegw1 + l * C * E,
            params.moeb1 + l * E,
            params.moew1 + l * E * C * C * 4,
            params.moeb1 + l * E * C * 4
        };

        SparseMoeParams moeGradients1{
            gradients.moeW1 + l * C * E,
            gradients.moeB1 + l * E,
            gradients.expertW1 + l * E * C * C * 4,
            gradients.expertB1 + l * E * C * 4
        };

        SparseMoeBackBuffers moeBackBuffers1{
            float* Xtmp1 + l * K * C,
            float* dL_dWtmp1 + l * K * C * C * 4,
            float* dL_dBtmp1 + l * K * C * 4,
            float* dE_dX1 + l * K * C,
            float* Wtmp1 + l * K * C * C * 4,
            float* Etmp1 + l * K * C * 4,
            float* EMA1 + l * E * C * 4, 
            float* J1 + l,
            float* dL_dG1 + l * E,
            float* dG_dX1 + l * K * C,
            float* dG_dZ1 + l * E,
            float* dL_dXtmp1 + l * K * C * 4,
            float* dL_dXsum1 + l * C
        };

        SparseMoeBuffers moeBuffers2{
            buffers.moeGate2 + l * (T - 1) * E,
            buffers.moeProbs2 + l * (T - 1) * E,
            buffers.moeSorted2 + l * (T - 1) * E,
            buffers.moeIndices2 + l * (T - 1) * E,
            buffers.moeW2 + l * K * (T - 1) * C * C * 4,
            buffers.moeB2 + l * K * (T - 1) * C,
            buffers.moeExperts2 + l * (T - 1) * K * C,
            buffers.moeY2 + l * (T - 1) * C 
        };

        SparseMoeParams moeParams2{
            params.moegw2 + l * C * 4 * E,
            params.moeb2 + l * E,
            params.moew2 + l * E * C * 4 * C,
            params.moeb2 + l * E * C
        };

        SparseMoeParams moeGradients2{
            gradients.moeW2 + l * C * 4 * E,
            gradients.moeB2 + l * E,
            gradients.expertW2 + l * E * C * 4 * C,
            gradients.expertB2 + l * E * C
        };

        SparseMoeBackBuffers moeBackBuffers2{
            float* Xtmp2 + l * C * 4,
            float* dL_dWtmp2 + l * K * C * 4 * C,
            float* dL_dBtmp2 + l * K * C,
            float* dE_dX2 + l * K * C * 4,
            float* Wtmp2 + l * K * C * 4 * C,
            float* Etmp2 + l * K * C,
            float* EMA2 + l * C, 
            float* J2 + l,
            float* dL_dG2 + l * E,
            float* dG_dX2 + l * K * C * 4,
            float* dG_dZ2 + l * E,
            float* dL_dXtmp2 + l * K * C * 4,
            float* dL_dXsum2 + l * C * 4
        };

        fusedSparseMoeBackward<<<moeGrid, moeBlock>>>(
            moeBuffers2, moeParams2, moeGradients2, moeBackBuffers2, moeBuffers1, moeParams1, moeGradients1, moeBackBuffers1,
            backbuffers.dL_dmoeY2 + l * C, backbuffers.dL_dmoeY1 + l * C * 4, backbuffers.dL_dmoeHidden + l * C * 4,
            backbuffers.loadBalance2 + l * E, backbuffers.loadBalance1 + l * E, gradients.projB + l * C,
             buffers.RMS + 2*l + 1, gradients.WNorm + (2* l + 1) * C * C, params.WNorm + (2* l + 1) * C * C, buffers.XNorm + (2*l + 1) * C, gradients.BNorm + (2*l + 1) * C, C, E, K, T, epsilon
        );
         

         AttentionBuffers attnBuffers{
            buffers.qkv + l * N * T * H * 3,
            buffers.Q + l * N * T * H,
            buffers.K + l * N * T * H,
            buffers.V + l * N * T * H,
            buffers.attnScores + l * N * T * T,
            buffers.W + l * N * T * H,
            buffers.attnY + l * N * T * H,
            buffers.projX+ l * T * C,
            buffers.projY + l * T * C
        };

         AttentionParams attnParams{
            params.qkvw + l * C * 3 * C,
            params.qkvb + l * 3 * C,
            params.projW + l * C * C,
            params.projB + l * C
        };

        AttentionParams attnGradients{
            gradients.qkvw + l * C * 3 * C,
            gradients.qkvb + l * 3 * C,
            gradients.projW + l * C * C,
            gradients.projB + l * C
        };

        AttentionBackBuffers attnBackBuffers{
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
            backbuffers.Qw + l * 3 * N * C * H,
            backbuffers.Kw + l * 3 * N * C * H,
            backbuffers.Vw + l * 3 * N * C * H,
            backbuffers.dL_dQCLS + l * C,
            backbuffers.dL_dKCLS + l * C,
            backbuffers.dL_dVCLS + l * C
        };

        if(l == 0){
            flashAttentionBackward<<<attnGrid, attnBlock>>>(
                attnBuffers, attnParams, attnGradients, attnBackBuffers,
                gradients.BNorm + 2*l * C, backbuffers.dL_dXNorm + l * C, gradients.CLS, buffers.XNorm + 2*l * C,
                buffers.attnX + l * C, buffers.RMS + 2*l, gradients.WNorm + 2*l * C * C, params.WNorm + 2*l * C * C, C, N, H, T, epsilon
            );
        }
        else{
            flashAttentionBackward<<<attnGrid, attnBlock>>>(
                attnBuffers, attnParams, attnGradients, attnBackBuffers,
                gradients.BNorm + 2*l * C, backbuffers.dL_dXNorm + l * C, backbuffers.dL_dmoeY + (l - 1) * C, buffers.XNorm + 2*l * C,
                buffers.attnX + l * C, buffers.RMS + 2*l, gradients.WNorm + 2*l * C * C, params.WNorm + 2*l * C * C, C, N, H, T, epsilon
            );
        }

    }

    dim3 optimGrid(C, 4*C, 4*C);
    dim3 optimBlock(4*C, 1, 1);
    optimizer<<<optimGrid, optimBlock>>>(params, gradients, velocity, momentum, config, learningRate, beta1, beta2, l2Lambda, time);
}



#endif