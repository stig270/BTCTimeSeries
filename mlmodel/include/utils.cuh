#include <cuda_runtime.h>

#ifndef UTILS_CUH
#define UTILS_CUH

__device__ inline int ceildiv(int a, int b){
    return (a + b - 1) / b;
}



//optimised
template<const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__device__ inline void sgemm(float* A, float*B, float*C, float alpha, float beta, int M, int N, int K, int D){
    const uint row = blockIdx.z;
    const uint col = blockIdx.y;

    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];

    const uint trow = threadIdx.x / (BN / TN);
    const uint tcol = threadIdx.x % (BN / TN);
    
    const uint resultsPerBlocktile = BM*BN;
    const uint threadsPerBlockTile = resultsPerBlocktile / (TM*TN);

    if(threadIdx.x < threadsPerBlockTile && row < ceildiv(M, TM) && col < ceildiv(N, TN) && blockIdx.x < D){
        
        A += row * K * BM + blockIdx.x * M * K;
        B += col * BN + blockIdx.x * K * N;
        C += row * N * BM + col * BN + blockIdx.x * M * N;
        
        const uint irowA = threadIdx.x / (BK / 4);
        const uint icolA = threadIdx.x % (BK / 4);
        const uint irowB = threadIdx.x / (BN / 4);
        const uint icolB = threadIdx.x % (BN / 4);
        


        float tresults[TM*TN] = {0.0};
        float regM[TM] = {0.0};
        float regN[TN] = {0.0};

        for(uint i = 0; i < K; ++i){

            float4 tmp = reinterpret_cast<float4*>(&A[irowA * K + icolA * 4])[0];
            As[(icolA * 4 + 0) * BM + irowA] = tmp.x;
            As[(icolA * 4 + 1) * BM + irowA] = tmp.y;
            As[(icolA * 4 + 2) * BM + irowA] = tmp.z;
            As[(icolA * 4 + 3) * BM + irowA] = tmp.w;   

            reinterpret_cast<float4*>(&Bs[icolB * 4 + irowB * BK])[0] = reinterpret_cast<float4*>(&B[irowB * N + icolB * 4])[0];
            __syncthreads();

            A += BK;
            B += BK * N;

            for(uint j = 0; j < BK; ++j){
                for(uint m = 0; m < TM; ++m){
                    regM[m] = As[j* BM + trow * TM + m];
                }
                for(uint n = 0; n < TN; ++n){
                    regN[n] = Bs[j * BN + tcol * TN + n];
                }
                for (uint m = 0; m < TM; ++m){
                    for (uint n = 0; n < TN; ++n){
                        tresults[m * TN + n] += regM[m] * regN[n];
                    }
                }
            }
            __syncthreads();

        }
        for (uint m = 0; m < TM; ++m){
            for (uint n = 0; n < TN; n += 4){
                float4 tmp = reinterpret_cast<float4*>(&C[(m + trow * TM) * N + tcol * TN + n])[0];
                tmp.x = alpha * tresults[m * TN + n] + beta * tmp.x;
                tmp.y = alpha * tresults[m * TN + n + 1] + beta * tmp.y;
                tmp.z = alpha * tresults[m * TN + n + 2] + beta * tmp.z;
                tmp.w = alpha * tresults[m * TN + n + 3] + beta * tmp.w;
                reinterpret_cast<float4*>(&C[(m + trow * TM) * N + tcol * TN + n])[0] = tmp;
                }
        }
        __syncthreads();
    }
}

//naive for testing
__device__ inline void sgemm_naive(float* A, float*B, float*C, float alpha, float beta, int M, int N, int K, int D){
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint col = blockIdx.y * blockDim.y + threadIdx.y;

    A += blockIdx.z * M * K;
    B += blockIdx.z * K * N;
    C += blockIdx.z * M * N;
    //enable 3d

    if(row < M &&  col < N){
        float value = 0.0;
        for(int i = 0; i < K; ++i){

            value += A[row * K + i] * B[i * N + col];
        }
        __syncthreads();
        C[row * N + col] = alpha * value + beta * C[row * N + col];
    }
}

__device__ inline void mergesort(float* X, float* Y, int* indices, int T, int E){
    
    int tid = threadIdx.x;
    int row = blockIdx.y;

    auto cas [](float &a, float &b, int& ia, int& ib){
        if(a < b){
            float tmp = a; a = b; b = tmp;
            int itmp = ia; ia = ib; ib = itmp;
        }

    
    };


    if(row < T && tid % T < E/4){
        float regX[4];
        reinterpret_cast<float4*>(regX)[0] = reinterpret_cast<float4*>(&X[row * E + tid * 4])[0];
        int regI[4];
        __syncthreads();

        for(int i = 0; i < 4; ++i){
            regI[i] = i;
        }

        cas(regX[0], regX[1], regI[0], regI[1]);
        cas(regX[2], regX[3], regI[2], regI[3]);
        

        #pragma unroll
        for(int i = 0; i < 2; ++i){
            if(regX[i] < regX[i + 2]){
                cas(regX[i], regX[i + 2], regI[i], regI[i + 2]);
                cas(regX[i + 1], regX[i + 2], regI[i + 1], regI[i + 2]);
            }
        }


        reinterpret_cast<float4*>(&Y[row * E + tid * 4])[0] = reinterpret_cast<float4*>(regX)[0];
        reinterpret_cast<int4*>(&indices[row * E + tid * 4])[0] = reinterpret_cast<int4*>(regI)[0];
        __syncThreads();


    }



}
// sort in descending order so we can iterate through k to get experts


__device__ inline void dotProduct(float* A, float* B, float*C, int D){
    uint vecPerRow = D / 4;
    uint tid = threadIdx.x;
    const uint mask = (1u << vecPerRow) - 1;
    float regA[4] = {0.0};
    float regB[4] = {0.0};
    float result = 0.0;
    for(uint i = 0; i < vecPerRow; ++i){
        reinterpret_cast<float4*>(regA)[0] = reinterpret_cast<float4*>(&A[tid * 4])[0];
        reinterpret_cast<float4*>(regB)[0] = reinterpret_cast<float4*>(&B[i * 4])[0];
        result += regA[0] * regB[0];
        result += regA[1] * regB[1];
        result += regA[2] * regB[2];
        result += regA[3] * regB[3];
    }

    for(uint offset = vecPerRow/2 ; offset > 0; offset >>= 1){
        result += __shfl_down_sync(mask, result, offset);
    }
    if(tid == 0){
        C[tid] = result;
    }
}


#endif 
