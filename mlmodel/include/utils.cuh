#ifndef UTILS_CUH
#define UTILS_CUH

__device__ inline int getGlobalIdx_3D_3D()
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) +
                   (threadIdx.z * (blockDim.x * blockDim.y)) +
                   (threadIdx.y * blockDim.x) +
                   threadIdx.x;
    return threadId;
}


#endif 