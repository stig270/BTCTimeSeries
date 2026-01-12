#ifndef SPARSEMOE_CUH
#define SPARSEMOE_CUH


#include "utils.cuh"
#include "linear.cuh"
#include "activationfunctions.cuh"
#include <cuda_runtime.h>


struct SparseMoeBuffers {
    float* gate;
    float* probs;
    float* sorted;
    int* indices;
    float* X;
    float* W;
    float* B;
    float* experts;
    float* Y;
};

struct SparseMoeParams {
    float* gw; 
    float* gb; 
    float* w; 
    float* b; 
};


struct SparseMoeBackBuffers{
    float* Xtmp1,
    float* dL_dWtmp,
    float* dL_dBtmp,
    float* dE_dX,
    float* Wtmp,
    float* Etmp,
    float* EMA, 
    float* J,
    float* dL_dG,
    float* dG_dX,
    float* dG_dZ,
    float* dL_dXtmp,
    float* dL_dXsum
};


__device__ inline void sparseMoeForward(float* X, const SparseMoeBuffers& buffers, const SparseMoeParams& params, int T, int Cin, int Cout, int E, int K){
    
    float* gate = buffers.gate;
    float* probs = buffers.probs;
    float* sorted = buffers.sorted;
    int* indices = buffers.indices;
    float* W = buffers.W;
    float* B = buffers.B;
    float* experts = buffers.experts;
    float* Y = buffers.Y;
    float* sum = buffers.sum;
    float* gw = params.gw;
    float* gb = params.gb;
    float* w = params.w;
    float* b = params.b;

   
    linearForward<Cin, 2, 8, 2, 1>(X, gw, gb, gate, T, E, Cin, 1);
    softmaxForward(gate, probs, T, E);
    mergesortForward(probs, sorted, indices, T, E);
    
    const uint tid = 4 * threadIdx.x;
    const uint token = blockIdx.x;
    const uint row = blockIdx.y;
    uint expertIdx;
    if(token < T && row < Cin){
        #pragma unroll
        for(int i = 0; i<K; ++i){
            expertIdx = indices[token * E + i];

            reinterpret_cast<float4*>(W[token * Cin * K * Cout + row * K * Cout + i * Cout + tid])[0] = reinterpret_cast<float4*>(&W[expertIdx * Cin * Cout + row * Cout + tid])[0];
            reinterpret_cast<float4*>(B[token * K * Cout + i * Cout * tid])[0] = reinterpret_cast<float4*>(&B[expertIdx * Cout + tid])[0];
            reinterpret_cast<float4*>(experts[token * K * Cout + i * Cout * tid])[0] = reinterpret_cast<float4*>(&B[token * K * Cout + i * Cout * tid])[0];
        } // double check this when GPU available 
    }

    linearForward<1, 32, 32, 1, 2>(X, W, B, experts,  1,  K * Cout, Cin, T);
        
    __shared__ float sortedShared[T * E];

    if(tid < E && token < T){
        reinterpret_cast<float4*>(sortedShared[token * E + tid])[0] = reinterpret_cast<float4*>(&sorted[token * E + tid])[0];
    
    #pragma unroll

    for(int i = 0; i < K; ++i){
            float4 regY = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 regExpert = reinterpret_cast<float4*>(&experts[token * K * Cout + i * Cout + tid])[0];
            float regProb = sortedShared[token * E + i];

            #pragma unroll
            for(int j = 0; j < 4; ++j){
                regY[j] += regExpert[j] * regProb;
            }

            reinterpret_cast<float4*>(&Y[token * Cout + tid])[0] = regY;
        }
    }
        __syncthreads();


    
}

//shared instead of global could be greatly beneficial
__device__ inline void sparseMoeBackward(float* dL_dY, float* dL_dX, float* X, const SparseMoeBuffers& buffers, const SparseMoeParams& params,
     SparseMoeParams& gradients, SparseMoeBackBuffers& backBuffers, float* loadBalance, int alpha, int beta, int T, int Cin, int Cout, int E, int K){
    
    const uint expert = blockIdx.x;
    const uint row = blockIdx.y;
    const uint col = blockIdx.z;
    const uint tid = threadIdx.x;   
    float expertIdx = buffers.sorted[expert];

    
    if(expert < K && tid < Cout/4){
        float gradY[4] = 0.0f;
        float gradB[4] = 0.0f;
        float gy = buffers.probs[expert];

        reinterpret_cast<float4*>(gradY)[0] = reinterpret_cast<float4*>(&dL_dY[4 * tid])[0];

        #pragma unroll
        for(int i = 0; i < 4; ++i){
            gradB[i] = gradY[i] * gy;
        }

        reinterpret_cast<float4*>(&backBuffers.dL_dBtmp[expert * Cout + 4 * tid])[0] = reinterpret_cast<float4*>(&gradB)[0];
        reinterpret_cast<float4*>(&gradients.b[expertIdx * Cout + 4 * tid])[0] = reinterpret_cast<float4*>(&gradB)[0];
                  
    
        __syncthreads();
    
        reinterpret_cast<float4*>(&backBuffers.Xtmp[expert * Cin + 4 * tid])[0] = reinterpret_cast<float4*>(&X[4 * tid])[0];
        reinterpret_cast<float4*>(&backBuffers.Wtmp[expert * Cin * Cout + row * Cout + 4 * tid])[0] = reinterpret_cast<float4*>(&params.w[expertIdx * Cin * Cout + row * Cout + 4 * tid])[0];
    
        linearBackward<1, 32, 1, 1, 4>(backBuffers.dL_dBtmp, backBuffers.dL_dWtmp, backBuffers.dE_dX, backBuffers.Xtmp, backBuffers.Wtmp, Cout, Cin, K);

        
        reinterpret_cast<float4*>(&gradients.w[expertIdx * Cin * Cout + row * Cout + 4 * tid])[0] = reinterpret_cast<float4*>(&backBuffers.Wtmp[expert * Cin * Cout + row * Cout + 4 * tid])[0];

        reinterpret_cast<float4*>(&backBuffers.Etmp[expert * Cout + 4 * tid])[0] = reinterpret_cast<float4*>(&buffers.experts[expertIdx * Cout + 4 * tid])[0];

        float ema[4] = 0.0f;
        float expEma[4] = 0.0f;
        reinterpret_cast<float4*>(&ema)[0] = reinterpret_cast<float4*>(&backBuffers.EMA[expertIdx * Cout + 4 * tid])[0];
        reinterpret_cast<float4*>(&expEma)[0] = reinterpret_cast<float4*>(&backBuffers.Etmp[expertIdx * Cout + 4 * tid])[0];

        #pragma unroll
        for(int i = 0; i < 4; ++i){
            ema[i] = beta * ema[i] + (1 - beta) * expEma[i];
        }
        reinterpret_cast<float4*>(&backBuffers.EMA[expertIdx * Cout + 4 * tid])[0] = reinterpret_cast<float4*>(&ema)[0];

        __syncthreads();
    
    }


    dotProduct(dL_dY, backBuffers.EMA, backBuffers.dL_dG, Cout, K);
    softmaxBackward(backBuffers.dL_dG, buffers.probs, backBuffers.J, gradients.gb, E, 1);
    sgemm<16, 4, 1, 4, 2>(gradients.gw, X, gradients.gb, 1.0f, 0.0f, Cin, E, 1, 1);

    if(expert == 0 && tid < E/4){
        float gb[4] = 0.0f;
        float lb[4] = 0.0f;
        reinterpret_cast<float4*>(&gb)[0] = reinterpret_cast<float4*>(&gradients.gb[4 * tid])[0];
        reinterpret_cast<float4*>(&lb)[0] = reinterpret_cast<float4*>(&loadBalance[4 * tid])[0];
        #pragma unroll
        for(int i = 0; i < 4; ++i){
            gb[i] += lb[i] * alpha;
        }

        reinterpret_cast<float4*>(&gradients.gb[4 * tid])[0] = reinterpret_cast<float4*>(&gb)[0];
        float gate[4] = 0.0f;
        float dG_dZ[K] = 0.0f;
        reinterpret_cast<float4*>(&gate)[0] = reinterpret_cast<float4*>(&buffers.probs[4 * tid])[0];
        for(int i = 0; i < K; ++i){
            dG_dZ[i] = gate[i] * (1 - gate[i]);
        }

    }
    
    if(expert < K && tid < Cin/4){
        float gw[4] = 0.0f;
        reinterpret_cast<float4*>(&gw)[0] = reinterpret_cast<float4*>(&params.gw[expertIdx * E + 4 * tid])[0];
        for(int i = 0; i < 4; ++i){
            gw[i] /= dG_dZ[expert];
        }
        reinterpret_cast<float4*>(&backBuffers.dG_dX[expert * Cin + 4 * tid])[0] = reinterpret_cast<float4*>(&gw)[0];
        

        float dE_dX[4] = 0.0f;
        reinterpret_cast<float4*>(&dE_dX)[0] = reinterpret_cast<float4*>(&backBuffers.dE_dX[expert * Cin + 4 * tid])[0];
        for(int i = 0; i < 4; ++i){
            dE_dX[i] = buffers.sorted[expertIdx] * dE_dX[i]
        }
    }
    sgemm<32, 32, 1, 4, 4>(backBuffers.dL_dXtmp, backBuffers.Etmp, backBuffers.dG_dX, 1.0f, 0.0f, Cout, Cin, 1, K);

    if(tid < Cin/4){
        float regdL_dXsum[4] = reinterpret_cast<float4*>(&dL_dXsum[4 * tid])[0];
        float regdL_dX[4] = 0.0f;

        for(int i = 0; i<K; ++i){
            float regdL_dXtmp[4] = reinterpret_cast<float4*>(&backBuffers.dL_dXtmp[i * Cin + 4 * tid])[0];
            float regdE_dX[4] = reinterpret_cast<float4*>(&backBuffers.dE_dX[i * Cin + 4 * tid])[0];

            #pragma unroll
            for(int j = 0; j < 4; ++j){
                regdL_dXsum[j] += regdL_dXtmp[j];
                regdL_dX[j] += regdE_dX[j];

            }


        }
        reinterpret_cast<float4*>(&dL_dXsum[4 * tid])[0] = reinterpret_cast<float4*>(&regdL_dXsum)[0];
        reinterpret_cast<float4*>(&dL_dX[4 * tid])[0] = reinterpret_cast<float4*>(&regdL_dX)[0];

    }


    sgemm<1, 32, 32, 1, 4>(dL_dX, dL_dY, backBuffers.dL_dXsum,  1.0f, 1.0f, 1, Cin, Cout, 1);
    
}
#endif