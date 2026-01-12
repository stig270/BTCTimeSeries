#ifndef ATTENTION_CUH
#define ATTENTION_CUH

#include "utils.cuh"
#include "linear.cu"
#include "activationfunctions.cuh"
#include <cuda_runtime.h>

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
    float* Qw; // qkw 
    float* Kw; // qkw + C*C
    float* Vw; // qkw + 2*C*C
    float* dL_dQCLS;
    float* dL_dKCLS;
    float* dL_dVCLS;
};

__device__ inline void attentionForward(float* X, const AttentionBuffers& buffers, const AttentionParams& params, int T, int C, int N, int H){
   
    float* qkvw = params.qkvw;
    float* qkvb = params.qkvb;
    float* projW = params.projW;
    float* projB = params.projB;
    float* qkv = buffers.qkv;
    float* Q = buffers.Q;
    float* K = buffers.K;
    float* V = buffers.V;
    float* KR = buffers.KR;
    float* QR = buffers.QR;
    float* KT = buffers.KT;
    float* scores = buffers.scores;
    float* W = buffers.W;
    float* Y = buffers.Y;
    float* projX = buffers.projX;
    float* projY = buffers.projY;

    
    
    linearForward<16, 16, 8, 4, 4>(X, qkvw, qkvb, qkv, T, 3 * N * H, C, 1);
    uint head = blockIdx.x;
    uint row = blockIdx.z;
    uint tid = threadIdx.x;
    uint colBase = tid * 4;
    uint strideQKV = 3 * N * H;

    if(tid < (H / 4) && head < N && row < T){
        

        reinterpret_cast<float4*>(&Q[(head * T + row) * H + colBase])[0] = reinterpret_cast<float4*>(&qkv[row * strideQKV + head * H + colBase])[0];
        reinterpret_cast<float4*>(&K[(head * T + row) * H + colBase])[0] = reinterpret_cast<float4*>(&qkv[row * strideQKV + N * H + head * H + colBase])[0];
        reinterpret_cast<float4*>(&V[(head * T + row) * H + colBase])[0] = reinterpret_cast<float4*>(&qkv[row * strideQKV + 2 * N * H + head * H + colBase])[0];


        float4 kVec = reinterpret_cast<float4*>(&K[(head * T + row) * H + colBase])[0];
        float4 qVec = reinterpret_cast<float4*>(&Q[(head * T + row) * H + colBase])[0];

        float thetaVal[2];
        float cosTheta[2];
        float sinTheta[2];

        // accesses could be further optimised to reduce bank conflicts
        reinterpret_cast<float2*>(&thetaVal)[0] = reinterpret_cast<float2*>(&buffers.theta[row * (H / 2) + tid * 2])[0];
        cosTheta[0] = cosf(thetaVal[0]);
        sinTheta[0] = sinf(thetaVal[0]);
        cosTheta[1] = cosf(thetaVal[1]);
        sinTheta[1] = sinf(thetaVal[1]);

        KR[(head * T + row) * H + colBase + 0] = kVec.x * cosTheta[0] - kVec.y * sinTheta[0];
        KR[(head * T + row) * H + colBase + 1] = kVec.x * sinTheta[0] + kVec.y * cosTheta[0];
        KR[(head * T + row) * H + colBase + 2] = kVec.z * cosTheta[1] - kVec.w * sinTheta[1];
        KR[(head * T + row) * H + colBase + 3] = kVec.z * sinTheta[1] + kVec.w * cosTheta[1];
        QR[(head * T + row) * H + colBase + 0] = qVec.x * cosTheta[0] - qVec.y * sinTheta[0];
        QR[(head * T + row) * H + colBase + 1] = qVec.x * sinTheta[0] + qVec.y * cosTheta[0];
        QR[(head * T + row) * H + colBase + 2] = qVec.z * cosTheta[1] - qVec.w * sinTheta[1];
        QR[(head * T + row) * H + colBase + 3] = qVec.z * sinTheta[1] + qVec.w * cosTheta[1];


        kVec = reinterpret_cast<float4*>(&KR[(head * T + row) * H + colBase])[0];

        __shared__ float Ks[H*T];
        if(colBase < H){
        Ks[colBase + 0 + row] = kVec.x;
        Ks[colBase + 1 + row] = kVec.y;
        Ks[colBase + 2 + row] = kVec.z;
        Ks[colBase + 3 + row] = kVec.w;
        }
        
        __syncthreads();

        if(colBase < H){
            #pragma unroll
            for(uint i = 0; i < 4; i++){
                KT[T * (head * H + colBase + i) + row] = Ks[colBase + i + row % H];
            }
        }   

        __syncthreads();
    }
    sgemm<16, 16, 4, 4, 4>(KT, QR, scores, rsqrtf((float)H), 0.0f, T, T, H, N);
    softmaxForward(scores, W, T, N);
    sgemm<16, 4, 16, 2, 2>(V, W, Y, 1.0f, 0.0f, T, H, T, N);

    if(tid < (H / 4) && head < N && row < T){
        reinterpret_cast<float4*>(&projX[(row * N + head) * H + colBase])[0] = 
            reinterpret_cast<float4*>(&Y[(head * T + row) * H + colBase])[0];

        __syncthreads();
    }


    linearForward<16, 16, 8, 4, 4>(projX, projW, projB, projY, T, C, C, 1);

}

__device__ inline void attentionBackward(const AttentionBuffers& buffers,
    const AttentionParams& params, AttentionParams& gradients, AttentionBackBuffers& backBuffers,
    float* X, float* dL_dX, int C, int N, int H, int T){

    linearBackward<16, 16, 8, 4, 4>(gradients.projB, gradients.projW, backBuffers.dL_dProjX, buffers.projX, params.projW, T, C, C, 1);
    

    const uint tid = threadIdx.x;
    const uint head = blockIdx.x;
    const uint col = blockIdx.y;
    const uint row = blockIdx.z;
    const uint colBase = tid * 4;
    if(tid < (H / 4) && head < N && row < T){
        reinterpret_cast<float4*>(&backBuffers.dL_dY[(head * T + row) * H + tid * 4])[0] = 
            reinterpret_cast<float4*>(&backBuffers.dL_dProjX[(row * N + head) * H + tid * 4])[0];

        __syncthreads();
    }

    sgemm<16, 2, 16, 4, 1>(buffers.W, backBuffers.dL_dY, backBuffers.dL_dV, 1.0f, 0.0f, T, H, T, N);
    sgemm<16, 16, 2, 4, 4>(backBuffers.dL_dY, buffers.V, backBuffers.dL_dW, 1.0f, 0.0f, T, T, H, N);


    softmaxBackward(backBuffers.dL_dW, buffers.W, backBuffers.J, backBuffers.dL_dScores, T, N);

    sgemm<16, 4, 16, 2, 4>(buffers.KT, backBuffers.dL_dScores, backBuffers.dL_dQR, rsqrtf((float)H), 0.0f, T, H, T, N);
    sgemm<4, 16, 16, 4, 2>(backBuffers.dL_dScores, buffers.QR, backBuffers.dL_dKT, rsqrtf((float)H), 0.0f, H, T, T, N);

    if(col < H && head < N && row < T/4){
        float4 dL_dKTvec = reinterpret_cast<float4*>(&backBuffers.dL_dKT[(head * T + col) * H + row * 4])[0];
        float4 dL_dKRvec;
        backBuffers.dL_dKR[(head * T + row) * H + col] = dL_dKTvec.x;
        backBuffers.dL_dKR[(head * T + row) * H + col + H] = dL_dKTvec.y;
        backBuffers.dL_dKR[(head * T + row) * H + col + 2 * H] = dL_dKTvec.z;
        backBuffers.dL_dKR[(head * T + row) * H + col + 3 * H] = dL_dKTvec.w;
    }

    if(tid < (H / 4) && head < N && row < T){
        float4 dL_dKRvec = reinterpret_cast<float4*>(&backBuffers.dL_dKR[(head * T + row) * H + colBase])[0];
        float4 dL_dQRvec = reinterpret_cast<float4*>(&backBuffers.dL_dQR[(head * T + row) * H + colBase])[0];

        float thetaVal[2];
        float cosTheta[2];
        float sinTheta[2];

        reinterpret_cast<float2*>(&thetaVal)[0] = reinterpret_cast<float2*>(&buffers.theta[row * (H / 2) + tid * 2])[0];
        cosTheta[0] = cosf(thetaVal[0]);
        sinTheta[0] = sinf(thetaVal[0]);
        cosTheta[1] = cosf(thetaVal[1]);
        sinTheta[1] = sinf(thetaVal[1]);

        backBuffers.dL_dK[(head * T + row) * H + colBase + 0] = dL_dKRvec.x * cosTheta[0] - dL_dKRvec.y * sinTheta[0];
        backBuffers.dL_dK[(head * T + row) * H + colBase + 1] = dL_dKRvec.x * sinTheta[0] + dL_dKRvec.y * cosTheta[0];
        backBuffers.dL_dK[(head * T + row) * H + colBase + 2] = dL_dKRvec.z * cosTheta[1] - dL_dKRvec.w * sinTheta[1];
        backBuffers.dL_dK[(head * T + row) * H + colBase + 3] = dL_dKRvec.z * sinTheta[1] + dL_dKRvec.w * cosTheta[1];
        backBuffers.dL_dQ[(head * T + row) * H + colBase + 0] = dL_dQRvec.x * cosTheta[0] - dL_dQRvec.y * sinTheta[0];
        backBuffers.dL_dQ[(head * T + row) * H + colBase + 1] = dL_dQRvec.x * sinTheta[0] + dL_dQRvec.y * cosTheta[0];
        backBuffers.dL_dQ[(head * T + row) * H + colBase + 2] = dL_dQRvec.z * cosTheta[1] - dL_dQRvec.w * sinTheta[1];
        backBuffers.dL_dQ[(head * T + row) * H + colBase + 3] = dL_dQRvec.z * sinTheta[1] + dL_dQRvec.w * cosTheta[1];

        __syncThreads();
    }


    const uint baseIdx = (head * H + tid)*4;
    if(tid < (H / 4) && head < N && row == 0){
        float4 dL_dQCLS = reinterpret_cast<float4*>(&backBuffers.dL_dQ[head * T * H + (T-1) * H + colBase])[0];
        float4 dL_dKCLS = reinterpret_cast<float4*>(&backBuffers.dL_dK[head * T * H + (T-1) * H + colBase])[0];
        float4 dL_dVCLS = reinterpret_cast<float4*>(&backBuffers.dL_dV[head * T * H + (T-1) * H + colBase])[0];


        reinterpret_cast<float4*>(&backBuffers.dL_dQCLS[head * H + colBase])[0] = dL_dQCLS;
        reinterpret_cast<float4*>(&backBuffers.dL_dKCLS[head * H + colBase])[0] = dL_dKCLS;
        reinterpret_cast<float4*>(&backBuffers.dL_dVCLS[head * H + colBase])[0] = dL_dVCLS;

        reinterpret_cast<float4*>(&gradients.qkvb[baseIdx])[0] = dL_dQCLS;
        reinterpret_cast<float4*>(&gradients.qkvb[baseIdx + N * H])[0] = dL_dKCLS;
        reinterpret_cast<float4*>(&gradients.qkvb[baseIdx + 2 * N * H])[0] = dL_dVCLS;
        __syncThreads();
    }

    sgemm<1, 16, 8, 1, 4>(X, gradients.qkvb, gradients.qkvw, 1.0f, 0.0f, C, 3*N*H, 1, 1);


    sgemm<1, 16, 8, 1, 4>(backBuffers.dL_dQCLS, backBuffers.Qw, dL_dX, 1.0f, 0.0f, 1, C, C, 1);
    sgemm<1, 16, 8, 1, 4>(backBuffers.dL_dKCLS, backBuffers.Kw, dL_dX, 1.0f, 1.0f, 1, C, C, 1);
    sgemm<1, 16, 8, 1, 4>(backBuffers.dL_dVCLS, backBuffers.Vw, dL_dX, 1.0f, 1.0f, 1, C, C, 1);

    


}



#endif