#ifndef OPTIMIZER_CUH
#define OPTIMIZER_CUH

#include "utils.cuh"
#include "config.h"
#include <cuda_runtime.h>

inline __device__ void optimizerStep(Parameter& t, float learningRate, float beta1, float beta2, float epsilon, float l2Lambda,
    int L, int E, int C, int T, int time){
    uint layer = blockIdx.x;
    uint expert = blockIdx.y;
    uint channel = blockIdx.z;
    uint tid = threadIdx.x;
    if(layer < L && expert < E && channel < C && tid < (T / 4)){
        if(T%4 == 0){
            uint idx = layer * E * C * T + expert * C * T + channel * T + tid * 4;
            float4 grad = reinterpret_cast<float4*>(t.gradient[idx])[0];
            float4 vel = reinterpret_cast<float4*>(t.velocity[idx])[0];
            float4 mom = reinterpret_cast<float4*>(t.momentum[idx])[0];

            mom.x = (beta1 * mom.x + (1 - beta1) * grad.x)/(1 - powf(beta1, time));
            mom.y = (beta1 * mom.y + (1 - beta1) * grad.y)/(1 - powf(beta1, time));
            mom.z = (beta1 * mom.z + (1 - beta1) * grad.z)/(1 - powf(beta1, time));
            mom.w = (beta1 * mom.w + (1 - beta1) * grad.w)/(1 - powf(beta1, time));

            vel.x = (beta2 * vel.x + (1 - beta2) * grad.x * grad.x)/(1 - powf(beta2, time));
            vel.y = (beta2 * vel.y + (1 - beta2) * grad.y * grad.y)/(1 - powf(beta2, time));
            vel.z = (beta2 * vel.z + (1 - beta2) * grad.z * grad.z)/(1 - powf(beta2, time));
            vel.w = (beta2 * vel.w + (1 - beta2) * grad.w * grad.w)/(1 - powf(beta2, time));

            reinterpret_cast<float4*>(t.velocity[idx])[0] = vel;
            reinterpret_cast<float4*>(t.momentum[idx])[0] = mom;

            float4 param = reinterpret_cast<float4*>(t.parameter[idx])[0];
            param.x -= learningRate * mom.x / (sqrtf(vel.x) + epsilon) + l2Lambda * param.x;
            param.y -= learningRate * mom.y / (sqrtf(vel.y) + epsilon) + l2Lambda * param.y;
            param.z -= learningRate * mom.z / (sqrtf(vel.z) + epsilon) + l2Lambda * param.z;
            param.w -= learningRate * mom.w / (sqrtf(vel.w) + epsilon) + l2Lambda * param.w;

            reinterpret_cast<float4*>(t.parameter[idx])[0] = param;
        }
        else{
            uint idx = layer * E * C * T + expert * C * T + channel * T + tid;
            t.velocity[idx] = beta1 * t.velocity[idx] + (1 - beta1) * t.gradient[idx];
            t.momentum[idx] = beta2 * t.momentum[idx] + (1 - beta2) * t.gradient[idx] * t.gradient[idx];
            t.parameter[idx] -= learningRate * t.velocity[idx] / (sqrtf(t.momentum[idx]) + epsilon) + l2Lambda * t.parameter[idx];
        }   
    }
}



#endif