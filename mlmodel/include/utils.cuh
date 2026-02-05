#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <cooperative_groups.h>

// Sense reversing barrier with atomic sychronization primatives
extern __device__ int SyncFlag;
extern __device__ int Count;

__device__ inline void await(){
    if(threadIdx.x == 0){
        
        int flag = atomicAdd(&SyncFlag, 0);
        int count = atomicAdd(&Count, 1) + 1;
        if(count == gridDim.x * gridDim.y * gridDim.z){
            atomicExch(&Count, 0);
            __threadfence();
            atomicExch(&SyncFlag, 1 - flag);
        }
        else{
            while(atomicAdd(&SyncFlag, 0) == flag){
                __nanosleep(32);
            }
            __threadfence();
        }
    }
    __syncthreads();
}


__device__ inline int ceildiv(int a, int b){
    return (a + b - 1) / b;
}



//optimised
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__device__ inline void sgemm(float* A, float*B, float*C, float alpha, float beta, int M, int N, int K, int D){
    const int row = blockIdx.z;
    const int col = blockIdx.y;

    
    const int trow = threadIdx.x / (BN / TN);
    const int tcol = threadIdx.x % (BN / TN);
    
    const int resultsPerBlocktile = BM*BN;
    const int threadsPerBlockTile = resultsPerBlocktile / (TM*TN);

    if(row < ceildiv(M, BM) && col < ceildiv(N, BN) && blockIdx.x < D){
        
        __shared__ float As[BM*BK];
        __shared__ float Bs[BK*BN];

        A += row * K * BM + blockIdx.x * M * K;
        B += col * BN + blockIdx.x * K * N;
        C += row * N * BM + col * BN + blockIdx.x * M * N;
        
        const int irowA = threadIdx.x / (BK / 4);
        const int icolA = threadIdx.x % (BK / 4);
        const int irowB = threadIdx.x / (BN / 4);
        const int icolB = threadIdx.x % (BN / 4);
        


        float tresults[TM*TN] = {0.0f};
        float regM[TM] = {0.0f};
        float regN[TN] = {0.0f};

        for(int i = 0; i < K; i += BK){

            if(threadIdx.x < BK * BM / 4){
                float4 tmp = reinterpret_cast<float4*>(&A[irowA * K + icolA * 4])[0];
                As[(icolA * 4 + 0) * BM + irowA] = tmp.x;
                As[(icolA * 4 + 1) * BM + irowA] = tmp.y;
                As[(icolA * 4 + 2) * BM + irowA] = tmp.z;
                As[(icolA * 4 + 3) * BM + irowA] = tmp.w;   
            }
            __syncthreads();
            if(threadIdx.x < BK * BN / 4){
                reinterpret_cast<float4*>(&Bs[icolB * 4 + irowB * BN])[0] = reinterpret_cast<float4*>(&B[irowB * N + icolB * 4])[0];
            }
            __syncthreads();
            if(threadIdx.x < threadsPerBlockTile){

                A += BK;
                B += BK * N;

                for(int j = 0; j < BK; ++j){
                    for(int m = 0; m < TM; ++m){
                        regM[m] = As[j* BM + trow * TM + m];
                    }
                    for(int n = 0; n < TN; ++n){
                        regN[n] = Bs[j * BN + tcol * TN + n];
                    }
                    for (int m = 0; m < TM; ++m){
                        for (int n = 0; n < TN; ++n){
                            tresults[m * TN + n] += regM[m] * regN[n];
                        }
                    }
                }
            }
            __syncthreads();

        }
        if(threadIdx.x < threadsPerBlockTile){

            for (int m = 0; m < TM; ++m){
                for (int n = 0; n < TN; n += 4){
                    float4 tmp = reinterpret_cast<float4*>(&C[(m + trow * TM) * N + tcol * TN + n])[0];
                    tmp.x = alpha * tresults[m * TN + n] + beta * tmp.x;
                    tmp.y = alpha * tresults[m * TN + n + 1] + beta * tmp.y;
                    tmp.z = alpha * tresults[m * TN + n + 2] + beta * tmp.z;
                    tmp.w = alpha * tresults[m * TN + n + 3] + beta * tmp.w;
                    reinterpret_cast<float4*>(&C[(m + trow * TM) * N + tcol * TN + n])[0] = tmp;
                }
            }
        }
        __syncthreads();
    }
    
    await();
    
}

//for testing
__device__ inline void sgemm_naive(float* A, float*B, float*C, float alpha, float beta, int M, int N, int K, int D){
    const int row = threadIdx.x;
    const int col = threadIdx.y;

    A += blockIdx.x * M * K;
    B += blockIdx.x * K * N;
    C += blockIdx.x * M * N;

    if(row < M &&  col < N && blockIdx.x < D){
        float value = 0.0f;
        for(int i = 0; i < K; ++i){

            value += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * value + beta * C[row * N + col];
        
    }
    __syncthreads();
    await();
    
}

__device__ inline void mergesort(float* X, float* Y, int* indices, int T, int E){
    
    int tid = threadIdx.x;
    int token = blockIdx.x;

    auto cas = [](float &a, float &b, int& ia, int& ib){
        if(a < b){
            float tmp = a; a = b; b = tmp;
            int itmp = ia; ia = ib; ib = itmp;
        }

    
    };


    if(blockIdx.z == 0 && blockIdx.y == 0 && token < T){
        if(tid < E/4){
            alignas(16) float regX[4];
            reinterpret_cast<float4*>(regX)[0] = reinterpret_cast<float4*>(&X[token * E + tid * 4])[0];
            alignas(16) int regI[4];

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


            reinterpret_cast<float4*>(&Y[token * E + tid * 4])[0] = reinterpret_cast<float4*>(regX)[0];
            reinterpret_cast<int4*>(&indices[token * E + tid * 4])[0] = reinterpret_cast<int4*>(regI)[0];
            
        }
        __syncthreads();
    }
    
    await();
    

}
// sort in descending order so we can iterate through k to get experts


__device__ inline void dotProduct(float* A, float* B, float*C, int D, int E, int F){
    int baseOffset = blockIdx.x * E * D + blockIdx.y * D;
    int vecPerRow = D / 4;
    int tid = threadIdx.x;
    const int mask = (1u << vecPerRow) - 1;
    float regA[4] = {0.0f};
    float regB[4] = {0.0f};
    float result = 0.0f;

    if(baseOffset < D * E * F){
        
        if(tid<vecPerRow){
           
            reinterpret_cast<float4*>(regA)[0] = reinterpret_cast<float4*>(&A[baseOffset + tid * 4])[0];
            reinterpret_cast<float4*>(regB)[0] = reinterpret_cast<float4*>(&B[baseOffset + tid * 4])[0];
            result += regA[0] * regB[0];
            result += regA[1] * regB[1];
            result += regA[2] * regB[2];
            result += regA[3] * regB[3];
            

            for(int offset = vecPerRow/2 ; offset > 0; offset >>= 1){
                result += __shfl_down_sync(mask, result, offset);
            }
            if(tid == 0){
                C[baseOffset] = result;
            }
        }

        __syncthreads();

    }
    await();

}


#endif 
