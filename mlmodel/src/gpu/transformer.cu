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
    
    //load balancing
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


void calculateThetaLauncher(float* X, float* theta, int T, int H, int B){
    dim3 gridDim(T, 1, 1);
    dim3 blockDim(H / 2, 1, 1);

    calculateTheta<<<gridDim, blockDim>>>(X, theta, T, H, B);
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

        if(l == 0){
            flashAttentionForward<<<attnGrid, attnBlock>>>(
            X, buffers.attnNorm[l], params.WNorm[2*l], params.BNorm[2*l],
            buffers.attnX[l],buffers.moeHiddenX1[l], attnBuffers, attnParams, T, C, N, H, epsilon
        );}
        else{
         
            flashAttentionForward<<<attnGrid, attnBlock>>>(
                buffers.attnNormX[l], buffers.attnNorm[l], params.WNorm[2*l], params.BNorm[2*l],
                buffers.attnX[l],buffers.moeHiddenX1[l], attnBuffers, attnParams, T, C, N, H, epsilon
            );
        }
        
      
        

        SparseMoeBuffers moeBuffers1{
            buffers.moeGate1[l],
            buffers.moeHiddenX1[l],
            buffers.moeProbs1[l],
            buffers.moeSorted1[l],
            buffers.moeIndices1[l],
            buffers.moeExperts1[l],
            buffers.moeY1[l]
        };

        SparseMoeParams moeParams1{
            params.moeW1[l],
            params.moeB1[l],
            params.expertW1[l],
            params.expertB1[l]
        };

        SparseMoeBuffers moeBuffers2{
            buffers.moeGate2[l],
            buffers.moeHiddenX2[l],
            buffers.moeProbs2[l],
            buffers.moeSorted2[l],
            buffers.moeIndices2[l],
            buffers.moeExperts2[l],
            buffers.moeY2[l]
        };

        SparseMoeParams moeParams2{
            params.moeW2[l],
            params.moeB2[l],
            params.expertW2[l],
            params.expertB2[l]
        };
        
        if(l < L - 1)
        fusedSparsemoeForward<<<moeGrid, moeBlock>>>(
            buffers.moeNormX[l], buffers.attnNormX[l + 1], buffers.moeNorm1[l], buffers.moeX[l], buffers.moeHidden[l], params.WNorm[2*l + 1], params.BNorm[2*l + 1],
             moeBuffers1, moeParams1, moeBuffers2, moeParams2, T, C, E, K, epsilon
        );
        else{
            fusedSparsemoeForward<<<moeGrid, moeBlock>>>(
            buffers.moeNormX[l], buffers.decoderY, buffers.moeNorm1[l], params.WNorm[2*l + 1], params.BNorm[2*l + 1],
            moeBuffers1, moeParams1, moeBuffers2, moeParams2, T, C, E, K, epsilon
            );
        }
            
    }

    // buffers.CLSNormX = buffers.decoderY + (T - 1) * C;
   

    dim3 gridDim(C, 1, 1);
    dim3 blockDim(C, 1, 1);

    classifierForward<<<gridDim, blockDim>>>(
        buffers.CLSNormX, buffers.CLSNorm, params.WNorm[2*L], params.BNorm[2*L],
        buffers.ZNormCLS, params.WCLS, params.BCLS, buffers.ZCLS, Y, C, F, epsilon
    );
    


}

void transformerBackward(ModelParameters& params, ModelParameters& gradients, ModelParameters& velocity, ModelParameters& momentum,
     const ModelHyperparameters& config, float learningRate, float beta1, float beta2, float l2Lambda, int time,
      IntermediateBuffers& buffers, BackpropagationBuffers& backbuffers, TrainingConfig& trainingConfig, float* Y){

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
        Y, buffers.Y, gradients.BCLS, gradients.WCLS, gradients.BNorm[2*L], buffers.ZNormCLS,
        params.WCLS, backbuffers.dL_dXNormCLS, buffers.CLSNormX, params.WNorm[2*L],
        gradients.WNorm[2*L], buffers.RMS[2*L], backbuffers.dL_moeY[L - 1], C, F, epsilon
    );


    dim3 moeGrid(E, 4*C, 4*C);
    dim3 moeBlock(C, 1, 1);


    dim3 attnGrid(N, C, T);
    dim3 attnBlock(C, 1, 1); 



    for(int l = L - 1; l >= 0; --l){


        SparseMoeBuffers moeBuffers1{
            buffers.moeGate1[l],
            buffers.moeHiddenX1[l],
            buffers.moeProbs1[l],
            buffers.moeSorted1[l],
            buffers.moeIndices1[l],
            buffers.moeExperts1[l],
            buffers.moeY1[l]
        };

        SparseMoeParams moeParams1{
            params.moeW1[l],
            params.moeB1[l],
            params.expertW1[l],
            params.expertB1[l]
        };

        SparseMoeParams moeGradients1{
            gradients.moeW1[l],
            gradients.moeB1[l],
            gradients.expertW1[l],
            gradients.expertB1[l]
        };

        SparseMoeBackBuffers moeBackBuffers1{
            float* Xtmp1[l],
            float* dL_dWtmp1[l],
            float* dL_dBtmp1[l],
            float* dE_dX1[l],
            float* Wtmp1[l],
            float* Etmp1[l],
            float* EMA1[l], 
            float* J1[l],
            float* dL_dG1[l],
            float* dG_dX1[l],
            float* dG_dZ1[l],
            float* dL_dXtmp1[l],
            float* dL_dXsum1[l]
        };

        SparseMoeBuffers moeBuffers2{
            buffers.moeGate2[l],
            buffers.moeHiddenX2[l],
            buffers.moeProbs2[l],
            buffers.moeSorted2[l],
            buffers.moeIndices2[l],
            buffers.moeExperts2[l],
            buffers.moeY2[l]
        };

        SparseMoeParams moeParams2{
            params.moeW2[l],
            params.moeB2[l],
            params.expertW2[l],
            params.expertB2[l]
        };

        SparseMoeParams moeGradients2{
            gradients.moeW2[l],
            gradients.moeB2[l],
            gradients.expertW2[l],
            gradients.expertB2[l]
        };

        SparseMoeBackBuffers moeBackBuffers2{
            float* Xtmp2[l],
            float* dL_dWtmp2[l],
            float* dL_dBtmp2[l],
            float* dE_dX2[l],
            float* Wtmp2[l],
            float* Etmp2[l],
            float* EMA2[l], 
            float* J2[l],
            float* dL_dG2[l],
            float* dG_dX2[l],
            float* dG_dZ2[l],
            float* dL_dXtmp2[l],
            float* dL_dXsum2[l]
        };

        sparseMoeBackward<<<moeGrid, moeBlock>>>(
            moeBuffers2, moeParams2, moeGradients2, moeBackBuffers2, moeBuffers1, moeParams1, moeGradients1, moeBackBuffers1,
            backbuffers.dL_dmoeY2[l], backbuffers.dL_dmoeY1[l], backbuffers.dL_dmoeHidden[l],
            backbuffers.loadBalance2[l], backbuffers.loadBalance1[l], backbuffers.dL_dX[l],
             buffers.RMS[2*l + 1], gradients.WNorm[2*l + 1], params.WNorm[2*l + 1], buffers.XNorm[2*l + 1], gradients.BNorm[2*l + 1], C, E, K, T, epsilon
        );
         

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

        AttentionParams attnGradients{
            gradients.qkvw[l],
            gradients.qkvb[l],
            gradients.projW[l],
            gradients.projB[l]
        };

        AttentionBackBuffers attnBackBuffers{
            backbuffers.dL_dProjX[l],
            backbuffers.dL_dY[l],
            backbuffers.dL_dW[l],
            backbuffers.dL_dV[l],
            backbuffers.dL_dK[l],
            backbuffers.dL_dQ[l],
            backbuffers.J[l],
            backbuffers.dL_dScores[l],
            backbuffers.dL_dKT[l],
            backbuffers.dL_dKR[l],
            backbuffers.dL_dQR[l],
            backbuffers.Qw[l],
            backbuffers.Kw[l],
            backbuffers.Vw[l],
            backbuffers.dL_dQCLS[l],
            backbuffers.dL_dKCLS[l],
            backbuffers.dL_dVCLS[l]
        };

        if(l == 0){
            flashAttentionBackward<<<attnGrid, attnBlock>>>(
                attnBuffers, attnParams, attnGradients, attnBackBuffers,
                backbuffers.dL_dAttnX[l], backbuffers.dL_dXNorm[l], gradients.CLS, buffers.XNorm[2*l],
                buffers.attnX[l], buffers.RMS[2*l], gradients.WNorm[2*l], params.WNorm[2*l], C, N, H, T, epsilon
            );
        }
        else{
            flashAttentionBackward<<<attnGrid, attnBlock>>>(
                attnBuffers, attnParams, attnGradients, attnBackBuffers,
                backbuffers.dL_dAttnX[l], backbuffers.dL_dXNorm[l], backbuffers.dL_dmoeY[l - 1], buffers.XNorm[2*l],
                buffers.attnX[l], buffers.RMS[2*l], gradients.WNorm[2*l], params.WNorm[2*l], C, N, H, T, epsilon
            );
        }

    }

    dim3 optimGrid(C, 4*C, 4*C);
    dim3 optimBlock(4*C, 1, 1);
    optimizer<<<optimGrid, optimBlock>>>(params, gradients, velocity, momentum, config, learningRate, beta1, beta2, l2Lambda, time);
}



#endif