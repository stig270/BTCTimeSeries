#ifndef SPARSEMOE_CUH
#define SPARSEMOE_CUH


#include "utils.cuh"
#include "linear.cuh"
#include "activationfunctions.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>


struct SparseMoeBuffers {
    float* gate;
    float* probs;
    float* sorted;
    int* indices;
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
    float* Xtmp;
    float* dL_dWtmp;
    float* dL_dBtmp;
    float* dE_dX;
    float* Wtmp;
    float* Etmp;
    float* EMA;
    float* J;
    float* dL_dG;
    float* dG_dX;
    float* dG_dZ;
    float* dL_dXtmp;
    float* dL_dXsum;

};


__device__ inline void sparseMoeForward(float* X, const SparseMoeBuffers& buffers, const SparseMoeParams& params,
     const int T, const int Cin, const int Cout, const int E, const int K){
    
    
    const int tid = 4 * threadIdx.x;
    const int token = blockIdx.x;
    int expertIdx;
    
    
    linearForward<16, 4, 16, 4, 4>(X, params.gw, params.gb, buffers.gate, T, E, Cin, 1);
    softmaxForward(buffers.gate, buffers.probs, E, 1, T);
    mergesort(buffers.probs, buffers.sorted, buffers.indices, T, E);
    
    if(token < T && blockIdx.y == 0 && blockIdx.z == 0){
        if(tid < Cout){
            #pragma unroll
            for(int i = 0; i<K; ++i){
                expertIdx = buffers.indices[token * E + i];
                for(int j = 0; j < Cin; ++j){
                    reinterpret_cast<float4*>(&buffers.W[token * Cin * K * Cout + j * K * Cout + i * Cout + tid])[0] = reinterpret_cast<float4*>(&params.w[expertIdx * Cin * Cout + j * Cout + tid])[0];
                }
                reinterpret_cast<float4*>(&buffers.B[token * K * Cout + i * Cout + tid])[0] = reinterpret_cast<float4*>(&params.b[expertIdx * Cout + tid])[0];
                reinterpret_cast<float4*>(&buffers.experts[token * K * Cout + i * Cout + tid])[0] = reinterpret_cast<float4*>(&buffers.B[token * K * Cout + i * Cout + tid])[0];
            }    
        }
        __syncthreads();    
    }
    await();
    
    if(Cout > Cin){
        linearForward<1, 128, 8, 1, 32>(X, buffers.W, buffers.B, buffers.experts,  1,  K * Cout, Cin, T);
    }
    else{
        linearForward<1, 64, 16, 1, 16>(X, buffers.W, buffers.B, buffers.experts,  1,  K * Cout, Cin, T);
    }    
    __shared__ float sortedShared[4];
    if(token < T && blockIdx.y == 0 && blockIdx.z == 0){
        if(tid == 0){
            reinterpret_cast<float4*>(&sortedShared[0])[0] = reinterpret_cast<float4*>(&buffers.sorted[token * E])[0];
        }
        __syncthreads();
        if(tid < Cout){
            float regY[4] = {0.0f};

            #pragma unroll
            for(int i = 0; i < K; ++i){
                float regProb = sortedShared[i];
                float regExpert[4] = {0.0f};
                reinterpret_cast<float4*>(&regExpert)[0] = reinterpret_cast<float4*>(&buffers.experts[token * K * Cout + i * Cout + tid])[0];
                

                #pragma unroll
                for(int j = 0; j < 4; ++j){
                    regY[j] += regExpert[j] * regProb;
                }
                
                
            }
            reinterpret_cast<float4*>(&buffers.Y[token * Cout + tid])[0] = reinterpret_cast<float4*>(&regY)[0];
        }
        __syncthreads();
    }
    await();

    
}

//shared instead of global could be greatly beneficial
__device__ inline void sparseMoeBackward(float* dL_dY, float* dL_dX, float* X, const SparseMoeBuffers& buffers, const SparseMoeParams& params,
     SparseMoeParams& gradients, SparseMoeBackBuffers& backBuffers, float* loadBalance,  const float alpha, const float beta,
      const int Cin, const int Cout, const int E, const int K){
    
    const int expert = blockIdx.x;
    const int tid = threadIdx.x;    
    const int expertIdx = buffers.indices[expert];

    
    if(expert < K && blockIdx.y == 0 && blockIdx.z == 0){
        if(tid < Cout/4 && threadIdx.y == 0){
            alignas(16) float gradY[4] = {0.0f};
            float gy = buffers.probs[expert];

            reinterpret_cast<float4*>(gradY)[0] = reinterpret_cast<float4*>(&dL_dY[4 * tid])[0];

            #pragma unroll
            for(int i = 0; i < 4; ++i){
                gradY[i] *= gy;
            }

            reinterpret_cast<float4*>(&backBuffers.dL_dBtmp[expert * Cout + 4 * tid])[0] = reinterpret_cast<float4*>(&gradY)[0];
            reinterpret_cast<float4*>(&gradients.b[expertIdx * Cout + 4 * tid])[0] = reinterpret_cast<float4*>(&gradY)[0];
                        
            for(int i = 0; i < Cin; ++i){
                reinterpret_cast<float4*>(&backBuffers.Wtmp[expert * Cin * Cout + i * Cout + 4 * tid])[0] =
                    reinterpret_cast<float4*>(&params.w[expertIdx * Cin * Cout + i * Cout + 4 * tid])[0];
            }
        }
        __syncthreads();
        if(tid<Cin/4 && threadIdx.y == 0){
            reinterpret_cast<float4*>(&backBuffers.Xtmp[expert * Cin + 4 * tid])[0] = reinterpret_cast<float4*>(&X[4 * tid])[0];
        }
        __syncthreads();
    }
    await();

    if(Cout > Cin){
        linearBackward<1, 64, 16, 1, 8, 4>(backBuffers.dL_dBtmp, backBuffers.dL_dWtmp, backBuffers.dE_dX, backBuffers.Xtmp, backBuffers.Wtmp, 1, Cout, Cin, K);
    }
    else{
        linearBackward<1, 16, 64, 1, 4, 8>(backBuffers.dL_dBtmp, backBuffers.dL_dWtmp, backBuffers.dE_dX, backBuffers.Xtmp, backBuffers.Wtmp, 1, Cout, Cin, K);
    }

    if(expert < K && blockIdx.y == 0 && blockIdx.z == 0){    
        if(tid < Cout/4 && threadIdx.y == 0){
            for(int i = 0; i < Cin; ++i){
                reinterpret_cast<float4*>(&gradients.w[expertIdx * Cin * Cout + i * Cout + 4 * tid])[0] = 
                    reinterpret_cast<float4*>(&backBuffers.dL_dWtmp[expert * Cin * Cout + i * Cout + 4 * tid])[0];
            }
            reinterpret_cast<float4*>(&backBuffers.Etmp[expert * Cout + 4 * tid])[0] = reinterpret_cast<float4*>(&buffers.experts[expertIdx * Cout + 4 * tid])[0];

            alignas(16) float ema[4] = {0.0f};
            alignas(16)float expEma[4] = {0.0f};
            reinterpret_cast<float4*>(&ema)[0] = reinterpret_cast<float4*>(&backBuffers.EMA[expertIdx * Cout + 4 * tid])[0];
            reinterpret_cast<float4*>(&expEma)[0] = reinterpret_cast<float4*>(&backBuffers.Etmp[expertIdx * Cout + 4 * tid])[0];

            #pragma unroll
            for(int i = 0; i < 4; ++i){
                ema[i] = beta * ema[i] + (1 - beta) * expEma[i];
            }
            reinterpret_cast<float4*>(&backBuffers.EMA[expertIdx * Cout + 4 * tid])[0] = reinterpret_cast<float4*>(&ema)[0];
        }
        __syncthreads();
    
    }
    await();


    sgemm<1, 4, 16, 1, 4>(dL_dY, backBuffers.EMA, backBuffers.dL_dG, 1.0f, 0.0f, 1, E, Cout, 1);
    softmaxBackward(backBuffers.dL_dG, buffers.probs, backBuffers.J, gradients.gb, E, 1, 1);

    if(expert == 0 && blockIdx.y == 0 && blockIdx.z == 0){
        if(tid < E/4 && threadIdx.y == 0){
            alignas(16) float gb[4] = {0.0f};
            alignas(16) float lb[4] = {0.0f};
            reinterpret_cast<float4*>(&gb)[0] = reinterpret_cast<float4*>(&gradients.gb[4 * tid])[0];
            reinterpret_cast<float4*>(&lb)[0] = reinterpret_cast<float4*>(&loadBalance[4 * tid])[0];
            #pragma unroll
            for(int i = 0; i < 4; ++i){
                gb[i] += lb[i] * alpha;
            }

            reinterpret_cast<float4*>(&gradients.gb[4 * tid])[0] = reinterpret_cast<float4*>(&gb)[0];
            alignas(16)float gate[4] = {0.0f};
            reinterpret_cast<float4*>(&gate)[0] = reinterpret_cast<float4*>(&buffers.probs[4 * tid])[0];
            for(int i = 0; i < E; ++i){
                gate[i] *= (1 - gate[i]);
            }
            reinterpret_cast<float4*>(&backBuffers.dG_dZ[4 * tid])[0] = reinterpret_cast<float4*>(&gate)[0];
        }
        __syncthreads();
    }
    await();

    outerProduct(X, gradients.gb, gradients.gw, Cin, E, 1, 1);
    
    if(expert < K && blockIdx.y == 0 && blockIdx.z == 0){
        if(tid < Cin/4 && threadIdx.y == 0){
            alignas(16) float gw[4] = {0.0f};
            float dG_dZ;
            reinterpret_cast<float4*>(&gw)[0] = reinterpret_cast<float4*>(&params.gw[expertIdx * Cin + 4 * tid])[0];
            dG_dZ= backBuffers.dG_dZ[expertIdx];
            for(int i = 0; i < 4; ++i){
                gw[i] *= dG_dZ;
            }
            reinterpret_cast<float4*>(&backBuffers.dG_dX[expert * Cin + 4 * tid])[0] = reinterpret_cast<float4*>(&gw)[0];
            
        }
        __syncthreads();
    }
    await();

    if(expert < K && blockIdx.y == 0 && blockIdx.z == 0){
        if(tid < Cin/4 && threadIdx.y == 0){
            alignas(16) float regdL_dX[4] = {0.0f};
            float regdL_dG = 0.0f;
            regdL_dG = backBuffers.dL_dG[expertIdx];
            reinterpret_cast<float4*>(&regdL_dX)[0] = reinterpret_cast<float4*>(&backBuffers.dG_dX[expert * Cin + 4 * tid])[0];
            #pragma unroll
            for(int i = 0; i < 4; ++i){
                regdL_dX[i] *= regdL_dG;
            }
            reinterpret_cast<float4*>(&backBuffers.dL_dXtmp[expert * Cin + 4 * tid])[0] = reinterpret_cast<float4*>(&regdL_dX)[0];
        }
        __syncthreads();
    }
    await();
    if(expert == 0 && blockIdx.y == 0 && blockIdx.z == 0){
        if(tid < Cin/4 && threadIdx.y == 0){
            alignas(16) float regdL_dX[4] = {0.0f};
            alignas(16) float regdL_dXtmp[4] = {0.0f};
            alignas(16) float regdE_dX[4] = {0.0f};
            for(int i = 0; i<K; ++i){
                
                reinterpret_cast<float4*>(&regdL_dXtmp)[0] = reinterpret_cast<float4*>(&backBuffers.dL_dXtmp[i * Cin + 4 * tid])[0];
                
                reinterpret_cast<float4*>(&regdE_dX)[0] = reinterpret_cast<float4*>(&backBuffers.dE_dX[i * Cin + 4 * tid])[0];

                #pragma unroll
                for(int j = 0; j < 4; ++j){
                    regdL_dX[j] += regdL_dXtmp[j] + regdE_dX[j];
                }


            }
            reinterpret_cast<float4*>(&dL_dX[4 * tid])[0] = reinterpret_cast<float4*>(&regdL_dX)[0];

        }
        __syncthreads();
    }
    await();

    
    
    
}
#endif