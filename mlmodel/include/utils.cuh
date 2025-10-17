#ifndef UTILS_CUH
#define UTILS_CUH

__device__ inline int getGlobalIdx()
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) +
                   (threadIdx.z * (blockDim.x * blockDim.y)) +
                   (threadIdx.y * blockDim.x) +
                   threadIdx.x;
    return threadId;
}

__device__ inline float treesum(float* vector, int size){
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size){
        shared[tid] = vector[idx];
    } else {
        shared[tid] = 0.0f;
    }
    __syncthreads();
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(tid < stride){
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();

    }
    __syncthreads();
    return shared[0];
}
#endif 