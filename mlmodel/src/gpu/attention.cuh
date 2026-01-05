#include "utils.cuh"
#include "linear.cu"
#include "activationfunctions.cuh"

#ifndef ATTENTION_CUH
#define ATTENTION_CUH

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
    uint rowBase = tid * 4;
    uint strideQKV = 3 * N * H;

    if(tid < (H / 4) && head < N && row < T){
        

        reinterpret_cast<float4*>(&Q[(head * T + row) * H + rowBase])[0] = reinterpret_cast<float4*>(&qkv[row * strideQKV + head * H + rowBase])[0];
        reinterpret_cast<float4*>(&K[(head * T + row) * H + rowBase])[0] = reinterpret_cast<float4*>(&qkv[row * strideQKV + N * H + head * H + rowBase])[0];
        reinterpret_cast<float4*>(&V[(head * T + row) * H + rowBase])[0] = reinterpret_cast<float4*>(&qkv[row * strideQKV + 2 * N * H + head * H + rowBase])[0];


        float4 kVec = reinterpret_cast<float4*>(&K[(head * T + row) * H + rowBase])[0];
        float4 qVec = reinterpret_cast<float4*>(&Q[(head * T + row) * H + rowBase])[0];

        float thetaVal[2];
        float cosTheta[2];
        float sinTheta[2];

        // accesses could be further optimised to reduce bank conflicts
        reinterpret_cast<float2*>(&thetaVal)[0] = reinterpret_cast<float2*>(&buffers.theta[row * (H / 2) + tid * 2])[0];
        cosTheta[0] = cosf(thetaVal[0]);
        sinTheta[0] = sinf(thetaVal[0]);
        cosTheta[1] = cosf(thetaVal[1]);
        sinTheta[1] = sinf(thetaVal[1]);

        KR[(head * T + row) * H + rowBase + 0] = kVec.x * cosTheta[0] - kVec.y * sinTheta[0];
        KR[(head * T + row) * H + rowBase + 1] = kVec.x * sinTheta[0] + kVec.y * cosTheta[0];
        KR[(head * T + row) * H + rowBase + 2] = kVec.z * cosTheta[1] - kVec.w * sinTheta[1];
        KR[(head * T + row) * H + rowBase + 3] = kVec.z * sinTheta[1] + kVec.w * cosTheta[1];
        QR[(head * T + row) * H + rowBase + 0] = qVec.x * cosTheta[0] - qVec.y * sinTheta[0];
        QR[(head * T + row) * H + rowBase + 1] = qVec.x * sinTheta[0] + qVec.y * cosTheta[0];
        QR[(head * T + row) * H + rowBase + 2] = qVec.z * cosTheta[1] - qVec.w * sinTheta[1];
        QR[(head * T + row) * H + rowBase + 3] = qVec.z * sinTheta[1] + qVec.w * cosTheta[1];


        kVec = reinterpret_cast<float4*>(&KR[(head * T + row) * H + rowBase])[0];

        __shared__ float Ks[H][T];
        if(rowBase < H){
        Ks[rowBase + 0][row] = kVec.x;
        Ks[rowBase + 1][row] = kVec.y;
        Ks[rowBase + 2][row] = kVec.z;
        Ks[rowBase + 3][row] = kVec.w;
        }
        
        __syncthreads();

        if(rowBase < H){
            #pragma unroll
            for(uint i = 0; i < 4; i++){
                KT[T * (head * H + rowBase + i) + row] = Ks[rowBase + i][row % H];
            }
        }   

        __syncthreads();
    }
    sgemm<16, 16, 4, 4, 4>(KT, QR, scores, rsqrtf((float)H), 0.0f, T, T, H, N);
    softmaxForward(scores, W, T, N);
    sgemm<16, 4, 16, 2, 2>(V, W, Y, 1.0f, 0.0f, T, H, T, N);

    if(tid < (H / 4) && head < N && row < T){
        reinterpret_cast<float4*>(&projX[(row * N + head) * H + rowBase])[0] = 
            reinterpret_cast<float4*>(&Y[(head * T + row) * H + rowBase])[0];

        __syncthreads();
    }
    linearForward<16, 16, 8, 4, 4>(projX, projW, projB, projY, T, C, C, 1);

}

__device__ inline void attentionBackward(float* dL_dY, const AttentionBuffers& buffers, const AttentionParams& params, AttentionParams& gradients,  int C, int N, int H){
}
#endif