#ifndef ATTENTION_CUH
#define ATTENTION_CUH

#include "utils.cuh"
#include "linear.cuh"
#include "activationfunctions.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

struct AttentionBuffers {
    float* theta;
    float* qkv;
    float* Q;
    float* K;
    float* V;
    float* QR;
    float* KR;
    float* KT;
    float* scores;
    float* W;
    float* Y;
    float* projX;
    float* projY;
};

struct AttentionParams {
    float* qkvw;
    float* qkvb;
    float* projW;
    float* projB;
};

struct AttentionBackBuffers {
    float* dL_dProjX;
    float* dL_dY;
    float* dL_dW;
    float* dL_dV;
    float* dL_dK;
    float* dL_dQ;
    float* J;
    float* dL_dScores;
    float* dL_dKT;
    float* dL_dKR;
    float* dL_dQR;
    float* dL_dqkv;
    float* dL_dXtmp;
};

__device__ inline void attentionForward(float* X, const AttentionBuffers& buffers, const AttentionParams& params,
     const int T, const int C, const int N, const int H){
   
    linearForward<16, 16, 8, 4, 4>(X, params.qkvw, params.qkvb, buffers.qkv, T, 3 * N * H, C, 1);
    
    int head = blockIdx.x;
    int row = blockIdx.z;
    int tid = threadIdx.x;
    int colBase = tid * 4;

    if(head < N && row < T && blockIdx.y == 0){
        
        if(colBase < H){
            float4 qVec = reinterpret_cast<float4*>(&buffers.qkv[(head * T + row) * H + colBase])[0];
            float4 kVec = reinterpret_cast<float4*>(&buffers.qkv[N*T*H + (head * T + row) * H + colBase])[0];
            reinterpret_cast<float4*>(&buffers.V[(head * T + row) * H + colBase])[0] = reinterpret_cast<float4*>(&buffers.qkv[2*N*T*H + (head * T + row) * H + colBase])[0];
            reinterpret_cast<float4*>(&buffers.Q[(head * T + row) * H + colBase])[0] = qVec;
            reinterpret_cast<float4*>(&buffers.K[(head * T + row) * H + colBase])[0] = kVec;
            float thetaVal[2];
            float cosTheta[2];
            float sinTheta[2];

            // accesses could be further optimised to reduce bank conflicts
            reinterpret_cast<float2*>(&thetaVal)[0] = reinterpret_cast<float2*>(&buffers.theta[row * (H / 2) + tid * 2])[0];
            cosTheta[0] = cosf(thetaVal[0]);
            sinTheta[0] = sinf(thetaVal[0]);
            cosTheta[1] = cosf(thetaVal[1]);
            sinTheta[1] = sinf(thetaVal[1]);

            alignas(16) float regKR[4] = {0.0f};
            alignas(16) float regQR[4] = {0.0f};

            regKR[0] = kVec.x * cosTheta[0] - kVec.y * sinTheta[0];
            regKR[1] = kVec.x * sinTheta[0] + kVec.y * cosTheta[0];
            regKR[2] = kVec.z * cosTheta[1] - kVec.w * sinTheta[1];
            regKR[3] = kVec.z * sinTheta[1] + kVec.w * cosTheta[1];
            regQR[0] = qVec.x * cosTheta[0] - qVec.y * sinTheta[0];
            regQR[1] = qVec.x * sinTheta[0] + qVec.y * cosTheta[0];
            regQR[2] = qVec.z * cosTheta[1] - qVec.w * sinTheta[1];
            regQR[3] = qVec.z * sinTheta[1] + qVec.w * cosTheta[1];

            reinterpret_cast<float4*>(&buffers.KR[(head * T + row) * H + colBase])[0] = reinterpret_cast<float4*>(regKR)[0];
            reinterpret_cast<float4*>(&buffers.QR[(head * T + row) * H + colBase])[0] = reinterpret_cast<float4*>(regQR)[0];

            
        
            #pragma unroll
            for(int i = 0; i < 4; i++){
                buffers.KT[T * (head * H + colBase + i) + row] = regKR[i];
            } 
        }
        __syncthreads();    
    }
    await();
   

    sgemm<16, 16, 4, 4, 4>(buffers.KT, buffers.QR, buffers.scores, rsqrtf((float)H), 0.0f, T, T, H, N);
    softmaxForward(buffers.scores, buffers.W, T, T, N);
    sgemm<16, 4, 16, 4, 4>(buffers.W, buffers.V, buffers.Y, 1.0f, 0.0f, T, H, T, N);

    if(head < N && row < T && blockIdx.y == 0){
        if(colBase < H){
        reinterpret_cast<float4*>(&buffers.projX[(row * N + head) * H + colBase])[0] = 
            reinterpret_cast<float4*>(&buffers.Y[(head * T + row) * H + colBase])[0];
        }
        __syncthreads();
    }
    await();

    linearForward<16, 16, 8, 4, 4>(buffers.projX, params.projW, params.projB, buffers.projY, T, C, C, 1);

}

__device__ inline void attentionBackward(const AttentionBuffers& buffers,
    const AttentionParams& params, AttentionParams& gradients, AttentionBackBuffers& backBuffers,
    float* X, float* dL_dX, const int C, const int N, const int H, const int T){

    linearBackward<16, 8, 16, 4, 4, 4>(gradients.projB, gradients.projW, backBuffers.dL_dProjX, buffers.projX, params.projW, T, C, C, 1);
    

    const int tid = threadIdx.x;
    const int head = blockIdx.x;
    const int col = blockIdx.y;
    const int row = blockIdx.z;
    const int colBase = tid * 4;
    if(head < N && row < T && blockIdx.y == 0){
        if(colBase < H && threadIdx.y == 0){
        reinterpret_cast<float4*>(&backBuffers.dL_dY[(head * T + row) * H + colBase])[0] = 
            reinterpret_cast<float4*>(&backBuffers.dL_dProjX[(row * N + head) * H + colBase])[0];
        }
        __syncthreads();
    }
    await();

    sgemm<16, 4, 16, 4, 4>(buffers.W, backBuffers.dL_dY, backBuffers.dL_dV, 1.0f, 0.0f, T, H, T, N);
    sgemm<16, 16, 4, 4, 4>(backBuffers.dL_dY, buffers.V, backBuffers.dL_dW, 1.0f, 0.0f, T, T, H, N);


    softmaxBackward(backBuffers.dL_dW, buffers.W, backBuffers.J, backBuffers.dL_dScores, T, T, N);

    sgemm<16, 4, 16, 4, 4>(buffers.KT, backBuffers.dL_dScores, backBuffers.dL_dQR, rsqrtf((float)H), 0.0f, T, H, T, N);
    sgemm<4, 16, 8, 4, 4>(backBuffers.dL_dScores, buffers.QR, backBuffers.dL_dKT, rsqrtf((float)H), 0.0f, H, T, T, N);

    if(blockIdx.y == 0 && head < N && row < T/4){
        if(colBase < H && threadIdx.y == 0){
            float4 dL_dKTvec = reinterpret_cast<float4*>(&backBuffers.dL_dKT[(head * H + tid) * T + row * 4])[0];
            float4 dL_dKRvec;
            backBuffers.dL_dKR[(head * T + row) * H + tid] = dL_dKTvec.x;
            backBuffers.dL_dKR[(head * T + row) * H + tid + H] = dL_dKTvec.y;
            backBuffers.dL_dKR[(head * T + row) * H + tid + 2 * H] = dL_dKTvec.z;
            backBuffers.dL_dKR[(head * T + row) * H + tid + 3 * H] = dL_dKTvec.w;
        
            float4 dL_dQRvec = reinterpret_cast<float4*>(&backBuffers.dL_dQR[(head * T + row) * H + colBase])[0];

            float thetaVal[2];
            float cosTheta[2];
            float sinTheta[2];

            reinterpret_cast<float2*>(&thetaVal)[0] = reinterpret_cast<float2*>(&buffers.theta[row * (H / 2) + tid * 2])[0];
            cosTheta[0] = cosf(thetaVal[0]);
            sinTheta[0] = sinf(thetaVal[0]);
            cosTheta[1] = cosf(thetaVal[1]);
            sinTheta[1] = sinf(thetaVal[1]);

            alignas(16) float regdL_dK[4] = {0.0f};
            alignas(16) float regdL_dQ[4] = {0.0f};
            regdL_dK[0] = dL_dKRvec.x * cosTheta[0] - dL_dKRvec.y * sinTheta[0];
            regdL_dK[1] = dL_dKRvec.x * sinTheta[0] + dL_dKRvec.y * cosTheta[0];
            regdL_dK[2] = dL_dKRvec.z * cosTheta[1] - dL_dKRvec.w * sinTheta[1];
            regdL_dK[3] = dL_dKRvec.z * sinTheta[1] + dL_dKRvec.w * cosTheta[1];
            regdL_dQ[0] = dL_dQRvec.x * cosTheta[0] - dL_dQRvec.y * sinTheta[0];
            regdL_dQ[1] = dL_dQRvec.x * sinTheta[0] + dL_dQRvec.y * cosTheta[0];
            regdL_dQ[2] = dL_dQRvec.z * cosTheta[1] - dL_dQRvec.w * sinTheta[1];
            regdL_dQ[3] = dL_dQRvec.z * sinTheta[1] + dL_dQRvec.w * cosTheta[1];

            reinterpret_cast<float4*>(&backBuffers.dL_dK[(head * T + row) * H + colBase])[0] = reinterpret_cast<float4*>(regdL_dK)[0];
            reinterpret_cast<float4*>(&backBuffers.dL_dQ[(head * T + row) * H + colBase])[0] = reinterpret_cast<float4*>(regdL_dQ)[0];
        }
        __syncthreads();
    }
    await();

    const int baseIdx = (head * H + tid)*4;
    if(blockIdx.y == 0 && head < N && row == 0){
        if(colBase < H && threadIdx.y == 0){
            float4 dL_dQCLS = reinterpret_cast<float4*>(&backBuffers.dL_dQ[head * T * H + (T-1) * H + colBase])[0];
            float4 dL_dKCLS = reinterpret_cast<float4*>(&backBuffers.dL_dK[head * T * H + (T-1) * H + colBase])[0];
            float4 dL_dVCLS = reinterpret_cast<float4*>(&backBuffers.dL_dV[head * T * H + (T-1) * H + colBase])[0];


            reinterpret_cast<float4*>(&backBuffers.dL_dqkv[head * H + colBase])[0] = dL_dQCLS;
            reinterpret_cast<float4*>(&backBuffers.dL_dqkv[head * H + colBase + N * H])[0] = dL_dKCLS;
            reinterpret_cast<float4*>(&backBuffers.dL_dqkv[head * H + colBase + 2 * N * H])[0] = dL_dVCLS;

            reinterpret_cast<float4*>(&gradients.qkvb[baseIdx])[0] = dL_dQCLS;
            reinterpret_cast<float4*>(&gradients.qkvb[baseIdx + N * H])[0] = dL_dKCLS;
            reinterpret_cast<float4*>(&gradients.qkvb[baseIdx + 2 * N * H])[0] = dL_dVCLS;
        }
        __syncthreads();
    }
    await();

    sgemm_naive(X, gradients.qkvb, gradients.qkvw, 1.0f, 0.0f, C, 3*N*H, 1, 1);


    sgemm<1, 16, 8, 1, 4>(backBuffers.dL_dqkv, params.qkvw, backBuffers.dL_dXtmp, 1.0f, 0.0f, 1, C, C, 3);
    
    if(blockIdx.y == 0 && row == 0 && head == 0){
        if(colBase < C && threadIdx.y == 0){
            alignas(16) float dL_dX1[4] = {0.0f};
            alignas(16) float dL_dX2[4] = {0.0f};
            alignas(16) float dL_dX3[4] = {0.0f};
            alignas(16) float regdL_dX[4] = {0.0f};
            reinterpret_cast<float4*>(&dL_dX1)[0] = reinterpret_cast<float4*>(&backBuffers.dL_dXtmp[tid * 4])[0];
            reinterpret_cast<float4*>(&dL_dX2)[0] = reinterpret_cast<float4*>(&backBuffers.dL_dXtmp[C + tid * 4])[0];
            reinterpret_cast<float4*>(&dL_dX3)[0] = reinterpret_cast<float4*>(&backBuffers.dL_dXtmp[2 * C + tid * 4])[0];
            #pragma unroll
            for(int i = 0; i < 4; i++){
                regdL_dX[i] = dL_dX1[i] + dL_dX2[i] + dL_dX3[i];
            }
            reinterpret_cast<float4*>(&dL_dX[tid * 4])[0] = reinterpret_cast<float4*>(&regdL_dX)[0];
        }
        __syncthreads();    
    }
    await();
    

}



#endif