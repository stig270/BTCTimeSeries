#include "utils.cuh"
#include "linear.cu"
#include "activationfunctions.cuh"

#ifndef ATTENTION_CUH
#define ATTENTION_CUH

struct AttentionBuffers {
    float* qkv;
    float* Q;
    float* K;
    float* V;
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


template<const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__device__ inline attentionForward(float* X, const AttentionBuffers& buffers, const AttentionParams& params, int T, int C, int N, int H){
   
    float* qkvw = params.qkvw;
    float* qkvb = params.qkvb;
    float* projW = params.projW;
    float* projB = params.projB;
    float* qkv = buffers.qkv;
    float* Q = buffers.Q;
    float* K = buffers.K;
    float* V = buffers.V;
    float* scores = buffers.scores;
    float* W = buffers.W;
    float* Y = buffers.Y;
    float* projX = buffers.projX;
    float* projY = buffers.projY;


    linearForward<BM, BN, BK, TM, TN>(X, qkvw, qkvb, qkv, T, 3 * N * H, C, 1);
    uint head = blockIdx.x;
    uint row = blockIdx.z;
    uint tid = threadIdx.x;
    uint rowBase = tid * 4;
    uint strideQKV = 3 * N * H;

    if(tid < (H / 4) && head < N && row < T){
        

        float4 qVec = reinterpret_cast<float4*>(&qkv[row * strideQKV + head * H + rowBase])[0];
        float4 kVec = reinterpret_cast<float4*>(&qkv[row * strideQKV + N * H + head * H + rowBase])[0];
        float4 vVec = reinterpret_cast<float4*>(&qkv[row * strideQKV + 2 * N * H + head * H + rowBase])[0];

        reinterpret_cast<float4*>(&Q[(head * T + row) * H + rowBase])[0] = qVec;
        reinterpret_cast<float4*>(&V[(head * T + row) * H + rowBase])[0] = vVec;

        __shared__ float Ks[H][T];
        if(rowBase < H){
        Ks[rowBase + 0][row] = kVec.x;
        Ks[rowBase + 1][row] = kVec.y;
        Ks[rowBase + 2][row] = kVec.z;
        Ks[rowBase + 3][row] = kVec.w;
        }
        
        __syncthreads();

        if(rowBase < H){
            for(uint i = 0; i < 4; i++){
                K[T * (head * H + rowBase + i) + row] = Ks[rowBase + i][row % H];
            }
        }   

        __syncthreads();
    }
    sgemm<BM, BN, BK, TM, TN>(K, Q, scores, rsqrtf((float)H), 0.0f, T, T, H, N);
    softmaxForward(scores, W, T, N);
    sgemm<BM, BN, BK, TM, TN>(V, W, Y, 1.0f, 0.0f, T, H, T, N);

    if(tid < (H / 4) && head < N && row < T){
        reinterpret_cast<float4*>(&projX[(row * N + head) * H + rowBase])[0] = 
            reinterpret_cast<float4*>(&Y[(head * T + row) * H + rowBase])[0];

        __syncthreads();
    }
    linearForward<BM, BN, BK, TM, TN>(projX, projW, projB, projY, T, C, C, 1);

}

#endif