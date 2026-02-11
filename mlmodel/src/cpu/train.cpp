#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include "config.h"
#include "learnratescheduler.h"
#include "model.h"
#include "dataprocessing.h"

#define CHECK_CUDA() checkCudaError(__FILE__, __LINE__)


void runTrainingLoop(const ModelHyperparameters& config, const TrainingConfig& trainConfig, const std::string& modelPath,
    TimeSeriesDataset& data, const std::vector<std::string>& train_timestamps, const std::vector<std::string>& val_timestamps,
     const bool initParameters){
    
    const float weightDecay = trainConfig.weightDecay;
    const float alpha = trainConfig.alpha;
    const float beta = trainConfig.beta;
    const float beta1 = trainConfig.beta1;
    const float beta2 = trainConfig.beta2;
    const float initialLR = trainConfig.initialLR;
    const float lrMin = trainConfig.lrMin;
    const int warmupSteps = trainConfig.warmupSteps;
    const int epochs = trainConfig.epochs;
    const int saveEvery = trainConfig.saveEvery;
    const int logEvery = trainConfig.logEvery;

    const int T = config.T;
    const int C = config.C;
    const int F = config.F;
    const int H = config.H;
    const int L = config.L;
    const int E = config.E;
    const int N = config.N;
    const int K = config.K;
    const int B = config.B;
    const float epsilon = config.epsilon;

    std::random_device rd;
    std::mt19937 gen(rd());

    ParameterSizes parameterSizes;
    parameterSizes.CLS = C;
    parameterSizes.WNorm = (2 * L + 1) * C * C;
    parameterSizes.BNorm = (2 * L + 1) * C;
    parameterSizes.qkvw = L * C * 3 * N * H;
    parameterSizes.qkvb = L * 3 * N * H;
    parameterSizes.projW = L * C * C;
    parameterSizes.projB = L * C;
    parameterSizes.moegw1 = L * C * E;
    parameterSizes.moegb1 = L * E;
    parameterSizes.moew1 = L * E * C * (C * 4);
    parameterSizes.moeb1 = L * E * (C * 4);
    parameterSizes.moegw2 = L * (C * 4) * E;
    parameterSizes.moegb2 = L * E;
    parameterSizes.moew2 = L * E * (C * 4) * C;
    parameterSizes.moeb2 = L * E * C;
    parameterSizes.WCLS = C * F;
    parameterSizes.BCLS = F;

    ModelParameters params;
    ModelParameters gradients;
    ModelParameters velocity;
    ModelParameters momentum;

    IntermediateBuffers buffers;  
    BackpropagationBuffers backbuffers;

    
    HostParameters hostParameters;
    if(initParameters){
        hostParameters = initializeParams(parameterSizes, config);

    }
    else{
        hostParameters = loadParams(parameterSizes, modelPath);
    }

    float* X;
    float* P;
    float* Y;
    float* loss;

    size_t XSize = T * C ; 
    size_t PSize = F;
    size_t YSize = F ;

    cudaMalloc(&X, XSize * sizeof(float));
    cudaMalloc(&P, PSize * sizeof(float));
    cudaMalloc(&Y, YSize * sizeof(float));
    cudaMalloc(&loss, sizeof(float));
    cudaMemset(loss, 0, sizeof(float));

    std::vector<float> XHost(XSize);
    std::vector<float> PHost(PSize);
    std::vector<float> YHost(YSize);
    float lossHost; 




    params.CLS = X + (T - 1) * C;
    cudaMemcpy(params.CLS, hostParameters.CLS.data(), parameterSizes.CLS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gradients.CLS, parameterSizes.CLS * sizeof(float));
    cudaMemset(gradients.CLS, 0, parameterSizes.CLS * sizeof(float));
    cudaMalloc(&velocity.CLS, parameterSizes.CLS * sizeof(float));
    cudaMemset(velocity.CLS, 0, parameterSizes.CLS * sizeof(float));
    cudaMalloc(&momentum.CLS, parameterSizes.CLS * sizeof(float));
    cudaMemset(momentum.CLS, 0, parameterSizes.CLS * sizeof(float));

    cudaMalloc(&params.WNorm, parameterSizes.WNorm * sizeof(float));
    cudaMemcpy(params.WNorm, hostParameters.WNorm.data(), parameterSizes.WNorm * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gradients.WNorm, parameterSizes.WNorm * sizeof(float));
    cudaMemset(gradients.WNorm, 0, parameterSizes.WNorm * sizeof(float));
    cudaMalloc(&velocity.WNorm, parameterSizes.WNorm * sizeof(float));
    cudaMemset(velocity.WNorm, 0, parameterSizes.WNorm * sizeof(float));
    cudaMalloc(&momentum.WNorm, parameterSizes.WNorm * sizeof(float));
    cudaMemset(momentum.WNorm, 0, parameterSizes.WNorm * sizeof(float));

    cudaMalloc(&params.BNorm, parameterSizes.BNorm * sizeof(float));
    cudaMemcpy(params.BNorm, hostParameters.BNorm.data(), parameterSizes.BNorm * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gradients.BNorm, parameterSizes.BNorm * sizeof(float));
    cudaMemset(gradients.BNorm, 0, parameterSizes.BNorm * sizeof(float));
    cudaMalloc(&velocity.BNorm, parameterSizes.BNorm * sizeof(float));
    cudaMemset(velocity.BNorm, 0, parameterSizes.BNorm * sizeof(float));
    cudaMalloc(&momentum.BNorm, parameterSizes.BNorm * sizeof(float));
    cudaMemset(momentum.BNorm, 0, parameterSizes.BNorm * sizeof(float));

    cudaMalloc(&params.qkvw, parameterSizes.qkvw * sizeof(float));
    cudaMemcpy(params.qkvw, hostParameters.qkvw.data(), parameterSizes.qkvw * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gradients.qkvw, parameterSizes.qkvw * sizeof(float));
    cudaMemset(gradients.qkvw, 0, parameterSizes.qkvw * sizeof(float));
    cudaMalloc(&velocity.qkvw, parameterSizes.qkvw * sizeof(float));
    cudaMemset(velocity.qkvw, 0, parameterSizes.qkvw * sizeof(float));
    cudaMalloc(&momentum.qkvw, parameterSizes.qkvw * sizeof(float));
    cudaMemset(momentum.qkvw, 0, parameterSizes.qkvw * sizeof(float));

    cudaMalloc(&params.qkvb, parameterSizes.qkvb * sizeof(float));
    cudaMemcpy(params.qkvb, hostParameters.qkvb.data(), parameterSizes.qkvb * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gradients.qkvb, parameterSizes.qkvb * sizeof(float));
    cudaMemset(gradients.qkvb, 0, parameterSizes.qkvb * sizeof(float));
    cudaMalloc(&velocity.qkvb, parameterSizes.qkvb * sizeof(float));
    cudaMemset(velocity.qkvb, 0, parameterSizes.qkvb * sizeof(float));
    cudaMalloc(&momentum.qkvb, parameterSizes.qkvb * sizeof(float));
    cudaMemset(momentum.qkvb, 0, parameterSizes.qkvb * sizeof(float));

    cudaMalloc(&params.projW, parameterSizes.projW * sizeof(float));
    cudaMemcpy(params.projW, hostParameters.projW.data(), parameterSizes.projW * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gradients.projW, parameterSizes.projW * sizeof(float));
    cudaMemset(gradients.projW, 0, parameterSizes.projW * sizeof(float));
    cudaMalloc(&velocity.projW, parameterSizes.projW * sizeof(float));
    cudaMemset(velocity.projW, 0, parameterSizes.projW * sizeof(float));
    cudaMalloc(&momentum.projW, parameterSizes.projW * sizeof(float));
    cudaMemset(momentum.projW, 0, parameterSizes.projW * sizeof(float));

    cudaMalloc(&params.projB, parameterSizes.projB * sizeof(float));
    cudaMemcpy(params.projB, hostParameters.projB.data(), parameterSizes.projB * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gradients.projB, parameterSizes.projB * sizeof(float));
    cudaMemset(gradients.projB, 0, parameterSizes.projB * sizeof(float));
    cudaMalloc(&velocity.projB, parameterSizes.projB * sizeof(float));
    cudaMemset(velocity.projB, 0, parameterSizes.projB * sizeof(float));
    cudaMalloc(&momentum.projB, parameterSizes.projB * sizeof(float));
    cudaMemset(momentum.projB, 0, parameterSizes.projB * sizeof(float));

    cudaMalloc(&params.moegw1, parameterSizes.moegw1 * sizeof(float));
    cudaMemcpy(params.moegw1, hostParameters.moegw1.data(), parameterSizes.moegw1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gradients.moegw1, parameterSizes.moegw1 * sizeof(float));
    cudaMemset(gradients.moegw1, 0, parameterSizes.moegw1 * sizeof(float));
    cudaMalloc(&velocity.moegw1, parameterSizes.moegw1 * sizeof(float));
    cudaMemset(velocity.moegw1, 0, parameterSizes.moegw1 * sizeof(float));
    cudaMalloc(&momentum.moegw1, parameterSizes.moegw1 * sizeof(float));
    cudaMemset(momentum.moegw1, 0, parameterSizes.moegw1 * sizeof(float));

    cudaMalloc(&params.moegb1, parameterSizes.moegb1 * sizeof(float));
    cudaMemcpy(params.moegb1, hostParameters.moegb1.data(), parameterSizes.moegb1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gradients.moegb1, parameterSizes.moegb1 * sizeof(float));
    cudaMemset(gradients.moegb1, 0, parameterSizes.moegb1 * sizeof(float));
    cudaMalloc(&velocity.moegb1, parameterSizes.moegb1 * sizeof(float));
    cudaMemset(velocity.moegb1, 0, parameterSizes.moegb1 * sizeof(float));
    cudaMalloc(&momentum.moegb1, parameterSizes.moegb1 * sizeof(float));
    cudaMemset(momentum.moegb1, 0, parameterSizes.moegb1 * sizeof(float));

    cudaMalloc(&params.moew1, parameterSizes.moew1 * sizeof(float));
    cudaMemcpy(params.moew1, hostParameters.moew1.data(), parameterSizes.moew1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gradients.moew1, parameterSizes.moew1 * sizeof(float));
    cudaMemset(gradients.moew1, 0, parameterSizes.moew1 * sizeof(float));
    cudaMalloc(&velocity.moew1, parameterSizes.moew1 * sizeof(float));
    cudaMemset(velocity.moew1, 0, parameterSizes.moew1 * sizeof(float));
    cudaMalloc(&momentum.moew1, parameterSizes.moew1 * sizeof(float));
    cudaMemset(momentum.moew1, 0, parameterSizes.moew1 * sizeof(float));

    cudaMalloc(&params.moeb1, parameterSizes.moeb1 * sizeof(float));
    cudaMemcpy(params.moeb1, hostParameters.moeb1.data(), parameterSizes.moeb1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gradients.moeb1, parameterSizes.moeb1 * sizeof(float));
    cudaMemset(gradients.moeb1, 0, parameterSizes.moeb1 * sizeof(float));
    cudaMalloc(&velocity.moeb1, parameterSizes.moeb1 * sizeof(float));
    cudaMemset(velocity.moeb1, 0, parameterSizes.moeb1 * sizeof(float));
    cudaMalloc(&momentum.moeb1, parameterSizes.moeb1 * sizeof(float));
    cudaMemset(momentum.moeb1, 0, parameterSizes.moeb1 * sizeof(float));

    cudaMalloc(&params.moegw2, parameterSizes.moegw2 * sizeof(float));
    cudaMemcpy(params.moegw2, hostParameters.moegw2.data(), parameterSizes.moegw2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gradients.moegw2, parameterSizes.moegw2 * sizeof(float));
    cudaMemset(gradients.moegw2, 0, parameterSizes.moegw2 * sizeof(float));
    cudaMalloc(&velocity.moegw2, parameterSizes.moegw2 * sizeof(float));
    cudaMemset(velocity.moegw2, 0, parameterSizes.moegw2 * sizeof(float));
    cudaMalloc(&momentum.moegw2, parameterSizes.moegw2 * sizeof(float));
    cudaMemset(momentum.moegw2, 0, parameterSizes.moegw2 * sizeof(float));

    cudaMalloc(&params.moegb2, parameterSizes.moegb2 * sizeof(float));
    cudaMemcpy(params.moegb2, hostParameters.moegb2.data(), parameterSizes.moegb2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gradients.moegb2, parameterSizes.moegb2 * sizeof(float));
    cudaMemset(gradients.moegb2, 0, parameterSizes.moegb2 * sizeof(float));
    cudaMalloc(&velocity.moegb2, parameterSizes.moegb2 * sizeof(float));
    cudaMemset(velocity.moegb2, 0, parameterSizes.moegb2 * sizeof(float));
    cudaMalloc(&momentum.moegb2, parameterSizes.moegb2 * sizeof(float));
    cudaMemset(momentum.moegb2, 0, parameterSizes.moegb2 * sizeof(float));

    cudaMalloc(&params.moew2, parameterSizes.moew2 * sizeof(float));
    cudaMemcpy(params.moew2, hostParameters.moew2.data(), parameterSizes.moew2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gradients.moew2, parameterSizes.moew2 * sizeof(float));
    cudaMemset(gradients.moew2, 0, parameterSizes.moew2 * sizeof(float));
    cudaMalloc(&velocity.moew2, parameterSizes.moew2 * sizeof(float));
    cudaMemset(velocity.moew2, 0, parameterSizes.moew2 * sizeof(float));
    cudaMalloc(&momentum.moew2, parameterSizes.moew2 * sizeof(float));
    cudaMemset(momentum.moew2, 0, parameterSizes.moew2 * sizeof(float));

    cudaMalloc(&params.moeb2, parameterSizes.moeb2 * sizeof(float));
    cudaMemcpy(params.moeb2, hostParameters.moeb2.data(), parameterSizes.moeb2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gradients.moeb2, parameterSizes.moeb2 * sizeof(float));
    cudaMemset(gradients.moeb2, 0, parameterSizes.moeb2 * sizeof(float));
    cudaMalloc(&velocity.moeb2, parameterSizes.moeb2 * sizeof(float));
    cudaMemset(velocity.moeb2, 0, parameterSizes.moeb2 * sizeof(float));
    cudaMalloc(&momentum.moeb2, parameterSizes.moeb2 * sizeof(float));
    cudaMemset(momentum.moeb2, 0, parameterSizes.moeb2 * sizeof(float));

    cudaMalloc(&params.WCLS, parameterSizes.WCLS * sizeof(float));
    cudaMemcpy(params.WCLS, hostParameters.WCLS.data(), parameterSizes.WCLS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gradients.WCLS, parameterSizes.WCLS * sizeof(float));
    cudaMemset(gradients.WCLS, 0, parameterSizes.WCLS * sizeof(float));
    cudaMalloc(&velocity.WCLS, parameterSizes.WCLS * sizeof(float));
    cudaMemset(velocity.WCLS, 0, parameterSizes.WCLS * sizeof(float));
    cudaMalloc(&momentum.WCLS, parameterSizes.WCLS * sizeof(float));
    cudaMemset(momentum.WCLS, 0, parameterSizes.WCLS * sizeof(float));

    cudaMalloc(&params.BCLS, parameterSizes.BCLS * sizeof(float));
    cudaMemcpy(params.BCLS, hostParameters.BCLS.data(), parameterSizes.BCLS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gradients.BCLS, parameterSizes.BCLS * sizeof(float));
    cudaMemset(gradients.BCLS, 0, parameterSizes.BCLS * sizeof(float));
    cudaMalloc(&velocity.BCLS, parameterSizes.BCLS * sizeof(float));
    cudaMemset(velocity.BCLS, 0, parameterSizes.BCLS * sizeof(float));
    cudaMalloc(&momentum.BCLS, parameterSizes.BCLS * sizeof(float));
    cudaMemset(momentum.BCLS, 0, parameterSizes.BCLS * sizeof(float));

    cudaMalloc(&buffers.theta, T * H / 2 * sizeof(float));
    cudaMemset(buffers.theta, 0, T * H / 2 * sizeof(float));
    cudaMalloc(&buffers.attnNormX, (L - 1) * T * C * sizeof(float));
    cudaMemset(buffers.attnNormX, 0, (L - 1) * T * C * sizeof(float));
    cudaMalloc(&buffers.attnNorm, L * T * C * sizeof(float));
    cudaMemset(buffers.attnNorm, 0, L * T * C * sizeof(float));
    cudaMalloc(&buffers.attnX, L * T * C * sizeof(float));
    cudaMemset(buffers.attnX, 0, L * T * C * sizeof(float));
    cudaMalloc(&buffers.RMS, (2*L + 1) * sizeof(float));
    cudaMemset(buffers.RMS, 0, (2*L + 1) * sizeof(float));
    cudaMalloc(&buffers.qkv, L * 3 * N * H * T * sizeof(float));
    cudaMemset(buffers.qkv, 0, L * 3 * N * H * T * sizeof(float));
    cudaMalloc(&buffers.Q, L * N * H * T * sizeof(float));
    cudaMemset(buffers.Q, 0, L * N * H * T * sizeof(float));
    cudaMalloc(&buffers.K, L * N * H * T * sizeof(float));
    cudaMemset(buffers.K, 0, L * N * H * T * sizeof(float));
    cudaMalloc(&buffers.V, L * N * H * T * sizeof(float));
    cudaMemset(buffers.V, 0, L * N * H * T * sizeof(float));
    cudaMalloc(&buffers.QR, L * N * H * T * sizeof(float));
    cudaMemset(buffers.QR, 0, L * N * H * T * sizeof(float));
    cudaMalloc(&buffers.KR, L * N * H * T * sizeof(float));
    cudaMemset(buffers.KR, 0, L * N * H * T * sizeof(float));
    cudaMalloc(&buffers.KT, L * N * H * T * sizeof(float));
    cudaMemset(buffers.KT, 0, L * N * H * T * sizeof(float));
    cudaMalloc(&buffers.attnScores, L * N * T * T * sizeof(float));
    cudaMemset(buffers.attnScores, 0, L * N * T * T * sizeof(float));
    cudaMalloc(&buffers.attnWeights, L * N * T * T * sizeof(float));
    cudaMemset(buffers.attnWeights, 0, L * N * T * T * sizeof(float));
    cudaMalloc(&buffers.attnY, L * T * C * sizeof(float));
    cudaMemset(buffers.attnY, 0, L * T * C * sizeof(float));
    cudaMalloc(&buffers.projX, L * T * C * sizeof(float));
    cudaMemset(buffers.projX, 0, L * T * C * sizeof(float));
    cudaMalloc(&buffers.projY, L * T * C * sizeof(float));
    cudaMemset(buffers.projY, 0, L * T * C * sizeof(float));
    cudaMalloc(&buffers.moeNormX, L * T * C * sizeof(float));
    cudaMemset(buffers.moeNormX, 0, L * T * C * sizeof(float));
    cudaMalloc(&buffers.moeNorm, L * T * C * sizeof(float));
    cudaMemset(buffers.moeNorm, 0, L * T * C * sizeof(float));
    cudaMalloc(&buffers.moeX, L * T * C * sizeof(float));
    cudaMemset(buffers.moeX, 0, L * T * C * sizeof(float));
    cudaMalloc(&buffers.moeGate1, L * T * E * sizeof(float));
    cudaMemset(buffers.moeGate1, 0, L * T * E * sizeof(float));
    cudaMalloc(&buffers.moeProbs1, L * T * E * sizeof(float));
    cudaMemset(buffers.moeProbs1, 0, L * T * E * sizeof(float));
    cudaMalloc(&buffers.moeSorted1, L * T * E * sizeof(float));
    cudaMemset(buffers.moeSorted1, 0, L * T * E * sizeof(float));
    cudaMalloc(&buffers.moeIndices1, L * T * E * sizeof(int));
    cudaMemset(buffers.moeIndices1, 0, L * T * E * sizeof(int));
    cudaMalloc(&buffers.moeW1, L * T * K * C * C * 4 *sizeof(float));
    cudaMemset(buffers.moeW1, 0, L * T * K * C * C * 4 *sizeof(float));
    cudaMalloc(&buffers.moeB1, L * T * K * C * 4 * sizeof(float));
    cudaMemset(buffers.moeB1, 0, L * T * K * C * 4 * sizeof(float));
    cudaMalloc(&buffers.moeExperts1, L * T * K * C * 4  * sizeof(float));
    cudaMemset(buffers.moeExperts1, 0, L * T * K * C * 4  * sizeof(float));
    cudaMalloc(&buffers.moeY1, L * T * C * 4 * sizeof(float));
    cudaMemset(buffers.moeY1, 0, L * T * C * 4 * sizeof(float));
    cudaMalloc(&buffers.moeHidden, L * T * C * 4 * sizeof(float));
    cudaMemset(buffers.moeHidden, 0, L * T * C * 4 * sizeof(float));
    cudaMalloc(&buffers.moeGate2, L * T * E * sizeof(float));
    cudaMemset(buffers.moeGate2, 0, L * T * E * sizeof(float));
    cudaMalloc(&buffers.moeProbs2, L * T * E * sizeof(float));
    cudaMemset(buffers.moeProbs2, 0, L * T * E * sizeof(float));
    cudaMalloc(&buffers.moeSorted2, L * T * E * sizeof(float));
    cudaMemset(buffers.moeSorted2, 0, L * T * E * sizeof(float));
    cudaMalloc(&buffers.moeIndices2, L * T * E * sizeof(int));
    cudaMemset(buffers.moeIndices2, 0, L * T * E * sizeof(int));
    cudaMalloc(&buffers.moeW2, L * T * K * C * 4 * C * sizeof(float));
    cudaMemset(buffers.moeW2, 0, L * T * K * C * 4 * C * sizeof(float));
    cudaMalloc(&buffers.moeB2, L * T * K * C * sizeof(float));
    cudaMemset(buffers.moeB2, 0, L * T * K * C * sizeof(float));
    cudaMalloc(&buffers.moeExperts2, L * T * K * C * sizeof(float));
    cudaMemset(buffers.moeExperts2, 0, L * T * K * C * sizeof(float));
    cudaMalloc(&buffers.moeY2, L * T * C * sizeof(float));
    cudaMemset(buffers.moeY2, 0, L * T * C * sizeof(float));
    cudaMalloc(&buffers.decoderY, T * C * sizeof(float));
    cudaMemset(buffers.decoderY, 0, T * C * sizeof(float));
    cudaMalloc(&buffers.CLSNorm, C * sizeof(float));
    cudaMemset(buffers.CLSNorm, 0, C * sizeof(float));
    cudaMalloc(&buffers.ZNormCLS, C * sizeof(float));
    cudaMemset(buffers.ZNormCLS, 0, C * sizeof(float));
    cudaMalloc(&buffers.ZCLS, F * sizeof(float)); 
    cudaMemset(buffers.ZCLS, 0, F * sizeof(float));
       

    cudaMalloc(&backbuffers.dL_dXNorm, (2 * L + 1) * C * sizeof(float));
    cudaMemset(backbuffers.dL_dXNorm, 0, (2 * L + 1) * C * sizeof(float));
    cudaMalloc(&backbuffers.dL_dXCLS, C * sizeof(float));
    cudaMemset(backbuffers.dL_dXCLS, 0, C * sizeof(float));
    cudaMalloc(&backbuffers.loadBalance2, L * E * sizeof(float));
    cudaMemset(backbuffers.loadBalance2, 0, L * E * sizeof(float));
    cudaMalloc(&backbuffers.dL_dMoeY2, L * C * sizeof(float));
    cudaMemset(backbuffers.dL_dMoeY2, 0, L * C * sizeof(float));
    cudaMalloc(&backbuffers.Xtmp2, L * K * C * 4 * sizeof(float));
    cudaMemset(backbuffers.Xtmp2, 0, L * K * C * 4 * sizeof(float));
    cudaMalloc(&backbuffers.dL_dWtmp2, L * K * C * 4 * C * sizeof(float));
    cudaMemset(backbuffers.dL_dWtmp2, 0, L * K * C * 4 * C * sizeof(float));
    cudaMalloc(&backbuffers.dL_dBtmp2, L * K * C * sizeof(float));
    cudaMemset(backbuffers.dL_dBtmp2, 0, L * K * C * sizeof(float));
    cudaMalloc(&backbuffers.dE_dX2, L * T * C * 4 * sizeof(float));
    cudaMemset(backbuffers.dE_dX2, 0, L * T * C * 4 * sizeof(float));
    cudaMalloc(&backbuffers.Wtmp2, L * K * C * 4 * C * sizeof(float));
    cudaMemset(backbuffers.Wtmp2, 0, L * K * C * 4 * C * sizeof(float));
    cudaMalloc(&backbuffers.Etmp2, L * K * C * sizeof(float));
    cudaMemset(backbuffers.Etmp2, 0, L * K * C * sizeof(float));
    cudaMalloc(&backbuffers.EMA2, L * E * C * sizeof(float));
    cudaMemset(backbuffers.EMA2, 0, L * E * C * sizeof(float));
    cudaMalloc(&backbuffers.J2, L * sizeof(float));
    cudaMemset(backbuffers.J2, 0, L * sizeof(float));
    cudaMalloc(&backbuffers.dL_dG2, L * E * sizeof(float));
    cudaMemset(backbuffers.dL_dG2, 0, L * E * sizeof(float));
    cudaMalloc(&backbuffers.dG_dX2, L * 4 * C * sizeof(float));
    cudaMemset(backbuffers.dG_dX2, 0, L * 4 * C * sizeof(float));
    cudaMalloc(&backbuffers.dG_dZ2, L * E * sizeof(float));
    cudaMemset(backbuffers.dG_dZ2, 0, L * E * sizeof(float));
    cudaMalloc(&backbuffers.dL_dXtmp2, L * K * C * 4 * sizeof(float));
    cudaMemset(backbuffers.dL_dXtmp2, 0, L * K * C * 4 * sizeof(float));
    cudaMalloc(&backbuffers.dL_dXsum2, L * C * 4 * sizeof(float));
    cudaMemset(backbuffers.dL_dXsum2, 0, L * C * 4 * sizeof(float));
    cudaMalloc(&backbuffers.dL_dMoeHidden, L * C * 4 * sizeof(float));
    cudaMemset(backbuffers.dL_dMoeHidden, 0, L * C * 4 * sizeof(float));
    cudaMalloc(&backbuffers.loadBalance1, L * E * sizeof(float));
    cudaMemset(backbuffers.loadBalance1, 0, L * E * sizeof(float));
    cudaMalloc(&backbuffers.dL_dMoeY1, L * T * C * 4 * sizeof(float));
    cudaMemset(backbuffers.dL_dMoeY1, 0, L * T * C * 4 * sizeof(float));
    cudaMalloc(&backbuffers.Xtmp1, L * T * C * sizeof(float));
    cudaMemset(backbuffers.Xtmp1, 0, L * T * C * sizeof(float));
    cudaMalloc(&backbuffers.dL_dWtmp1, L * K * C * C * 4 * sizeof(float));
    cudaMemset(backbuffers.dL_dWtmp1, 0, L * K * C * C * 4 * sizeof(float));
    cudaMalloc(&backbuffers.dL_dBtmp1, L * K * C * 4 * sizeof(float));
    cudaMemset(backbuffers.dL_dBtmp1, 0, L * K * C * 4 * sizeof(float));
    cudaMalloc(&backbuffers.dE_dX1, L * T * C * sizeof(float));
    cudaMemset(backbuffers.dE_dX1, 0, L * T * C * sizeof(float));
    cudaMalloc(&backbuffers.Wtmp1, L * K * C * C * 4 * sizeof(float));
    cudaMemset(backbuffers.Wtmp1, 0, L * K * C * C * 4 * sizeof(float));
    cudaMalloc(&backbuffers.Etmp1, L * K * C * 4 * sizeof(float));
    cudaMemset(backbuffers.Etmp1, 0, L * K * C * 4 * sizeof(float));
    cudaMalloc(&backbuffers.EMA1, L * E * C * 4 * sizeof(float));
    cudaMemset(backbuffers.EMA1, 0, L * E * C * 4 * sizeof(float));
    cudaMalloc(&backbuffers.J1, L * sizeof(float));
    cudaMemset(backbuffers.J1, 0, L * sizeof(float));
    cudaMalloc(&backbuffers.dL_dG1, L * E * sizeof(float));
    cudaMemset(backbuffers.dL_dG1, 0, L * E * sizeof(float));
    cudaMalloc(&backbuffers.dG_dX1, L * C * sizeof(float));
    cudaMemset(backbuffers.dG_dX1, 0, L * C * sizeof(float));
    cudaMalloc(&backbuffers.dG_dZ1, L * E * sizeof(float));
    cudaMemset(backbuffers.dG_dZ1, 0, L * E * sizeof(float));
    cudaMalloc(&backbuffers.dL_dXtmp1, L * K * C * sizeof(float));
    cudaMemset(backbuffers.dL_dXtmp1, 0, L * K * C * sizeof(float));
    cudaMalloc(&backbuffers.dL_dXsum1, L * C * sizeof(float));
    cudaMemset(backbuffers.dL_dXsum1, 0, L * C * sizeof(float));
    cudaMalloc(&backbuffers.dL_dProjX, L * T * C * sizeof(float));
    cudaMemset(backbuffers.dL_dProjX, 0, L * T * C * sizeof(float));
    cudaMalloc(&backbuffers.dL_dY, L * T * C * sizeof(float));
    cudaMemset(backbuffers.dL_dY, 0, L * T * C * sizeof(float));
    cudaMalloc(&backbuffers.dL_dW, L * N * T * T * sizeof(float));
    cudaMemset(backbuffers.dL_dW, 0, L * N * T * T * sizeof(float));
    cudaMalloc(&backbuffers.dL_dK, L * N * H * T * sizeof(float));
    cudaMemset(backbuffers.dL_dK, 0, L * N * H * T * sizeof(float));
    cudaMalloc(&backbuffers.dL_dQ, L * N * H * T * sizeof(float));
    cudaMemset(backbuffers.dL_dQ, 0, L * N * H * T * sizeof(float));
    cudaMalloc(&backbuffers.dL_dV, L * N * H * T * sizeof(float));
    cudaMemset(backbuffers.dL_dV, 0, L * N * H * T * sizeof(float));
    cudaMalloc(&backbuffers.J, L * N * T * sizeof(float));
    cudaMemset(backbuffers.J, 0, L * N * T * sizeof(float));
    cudaMalloc(&backbuffers.dL_dScores, L * N * T * T * sizeof(float));
    cudaMemset(backbuffers.dL_dScores, 0, L * N * T * T * sizeof(float));
    cudaMalloc(&backbuffers.dL_dKT, L * N * H * T * sizeof(float));
    cudaMemset(backbuffers.dL_dKT, 0, L * N * H * T * sizeof(float));
    cudaMalloc(&backbuffers.dL_dQR, L * N * H * T * sizeof(float));
    cudaMemset(backbuffers.dL_dQR, 0, L * N * H * T * sizeof(float));
    cudaMalloc(&backbuffers.dL_dKR, L * N * H * T * sizeof(float));
    cudaMemset(backbuffers.dL_dKR, 0, L * N * H * T * sizeof(float));
    cudaMalloc(&backbuffers.dL_dqkv, L * C * 3 * sizeof(float));
    cudaMemset(backbuffers.dL_dqkv, 0, L * C * 3 * sizeof(float));
    cudaMalloc(&backbuffers.dL_dXtmp, L * C * 3 * sizeof(float));
    cudaMemset(backbuffers.dL_dXtmp, 0, L * C * 3 * sizeof(float));
    
    int t = 0;
    float learnRate;
    
    calculateThetaLauncher(buffers.theta, T, H, B);


    for (int e = 0; e < epochs; e++) {
       
        auto epochStart = std::chrono::high_resolution_clock::now();
        std::vector<std::string> shuffled_timestamps = train_timestamps;
        std::shuffle(shuffled_timestamps.begin(), shuffled_timestamps.end(), gen);
        float trainAccuracy = 0.0f;
        float valAccuracy = 0.0f;

        for(size_t i = 0; i < shuffled_timestamps.size(); i++){
            const std::string& timestamp = shuffled_timestamps[i];
            std::vector<std::vector<float>> window = data.getWindow(timestamp, T);
            float label = data.getLabel(timestamp, T);

            for(int j = 0; j < T - 1; j++){
                for(int c = 0; c < C; c++){
                    XHost[j * C + c] = window[j][c];
                }
            }

            for(int f = 0; f < F; f++){
                YHost[f] = (f == static_cast<int>(label)) ? 1.0f : 0.0f;
            }

            cudaMemcpy(X, XHost.data(), (XSize - C) * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(Y, YHost.data(), YSize * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(P, 0, PSize * sizeof(float));

            
            learnRate = calculateLearnRate(initialLR, lrMin,  warmupSteps, t, epochs * shuffled_timestamps.size());
            transformerForward(X, P, params, config, buffers);
            cudaDeviceSynchronize();
            transformerBackward(params, gradients, velocity, momentum, config, learnRate, alpha, beta, beta1, beta2, weightDecay, t,
                buffers, backbuffers, Y, P);
            CHECK_CUDA();
            cudaDeviceSynchronize();
            cudaMemcpy(&PHost, P, PSize * sizeof(float), cudaMemcpyDeviceToHost); 
            if(PHost[static_cast<int>(label)] > PHost[1 - static_cast<int>(label)]){
                trainAccuracy += 1.0f;
            }

            if (t % logEvery == 0) {
                calculateLoss(P, Y, loss, F, epsilon);
                cudaMemcpy(&lossHost, loss, sizeof(float), cudaMemcpyDeviceToHost);
                std::cout << "Step " << t << ", Loss: " << lossHost << std::endl;
                cudaMemset(loss, 0, sizeof(float));
            }

            if (t % saveEvery == 0) {
                cudaMemcpy(hostParameters.CLS.data(), params.CLS, parameterSizes.CLS * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostParameters.WNorm.data(), params.WNorm, parameterSizes.WNorm * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostParameters.BNorm.data(), params.BNorm, parameterSizes.BNorm * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostParameters.qkvw.data(), params.qkvw, parameterSizes.qkvw * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostParameters.qkvb.data(), params.qkvb, parameterSizes.qkvb * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostParameters.projW.data(), params.projW, parameterSizes.projW * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostParameters.projB.data(), params.projB, parameterSizes.projB * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostParameters.moegw1.data(), params.moegw1, parameterSizes.moegw1 * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostParameters.moegb1.data(), params.moegb1, parameterSizes.moegb1 * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostParameters.moew1.data(), params.moew1, parameterSizes.moew1 * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostParameters.moeb1.data(), params.moeb1, parameterSizes.moeb1 * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostParameters.moegw2.data(), params.moegw2, parameterSizes.moegw2 * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostParameters.moegb2.data(), params.moegb2, parameterSizes.moegb2 * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostParameters.moew2.data(), params.moew2, parameterSizes.moew2 * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostParameters.moeb2.data(), params.moeb2, parameterSizes.moeb2 * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostParameters.WCLS.data(), params.WCLS, parameterSizes.WCLS * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostParameters.BCLS.data(), params.BCLS, parameterSizes.BCLS * sizeof(float), cudaMemcpyDeviceToHost);

                saveParams(parameterSizes, hostParameters, modelPath);
            }

            t++;
        }

        trainAccuracy /= shuffled_timestamps.size();

        shuffled_timestamps = val_timestamps;
        std::shuffle(shuffled_timestamps.begin(), shuffled_timestamps.end(), gen);
        for(size_t i = 0; i < shuffled_timestamps.size(); i++){

            const std::string& timestamp = shuffled_timestamps[i];
            std::vector<std::vector<float>> window = data.getWindow(timestamp, T);
            float label = data.getLabel(timestamp, T);

            for(int j = 0; j < T - 1; j++){
                for(int c = 0; c < C; c++){
                    XHost[j * C + c] = window[j][c];
                }
            }
            cudaMemset(P, 0, PSize * sizeof(float));
            cudaMemcpy(X, XHost.data(), (XSize - C) * sizeof(float), cudaMemcpyHostToDevice);
            transformerForward(X, P, params, config, buffers);
            cudaMemcpy(&PHost, P, PSize * sizeof(float), cudaMemcpyDeviceToHost);

            if(PHost[static_cast<int>(label)] > PHost[1 - static_cast<int>(label)]){
                valAccuracy += 1.0f;
            }
    

        }   

        valAccuracy /= shuffled_timestamps.size();
        auto epochEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epochDuration = epochEnd - epochStart;
        std::cout << "Epoch " << e << " completed in " << epochDuration.count() << " seconds" << std::endl;
        std::cout << "Training Accuracy: " << trainAccuracy * 100 << "%" << std::endl;
        std::cout << "Validation Accuracy: " << valAccuracy * 100 << "%" << std::endl;

    }
    cudaFree(X);
    cudaFree(P);
    cudaFree(Y);
    cudaFree(loss);

    cudaFree(params.WNorm);
    cudaFree(params.BNorm);
    cudaFree(params.qkvw);
    cudaFree(params.qkvb);
    cudaFree(params.projW);
    cudaFree(params.projB);
    cudaFree(params.moegw1);
    cudaFree(params.moegb1);
    cudaFree(params.moew1);
    cudaFree(params.moeb1);
    cudaFree(params.moegw2);
    cudaFree(params.moegb2);
    cudaFree(params.moew2);
    cudaFree(params.moeb2);
    cudaFree(params.WCLS);
    cudaFree(params.BCLS);

    cudaFree(gradients.CLS);
    cudaFree(gradients.WNorm);
    cudaFree(gradients.BNorm);
    cudaFree(gradients.qkvw);
    cudaFree(gradients.qkvb);
    cudaFree(gradients.projW);
    cudaFree(gradients.projB);
    cudaFree(gradients.moegw1);
    cudaFree(gradients.moegb1);
    cudaFree(gradients.moew1);
    cudaFree(gradients.moeb1);
    cudaFree(gradients.moegw2);
    cudaFree(gradients.moegb2);
    cudaFree(gradients.moew2);
    cudaFree(gradients.moeb2);
    cudaFree(gradients.WCLS);
    cudaFree(gradients.BCLS);

    cudaFree(velocity.CLS);
    cudaFree(velocity.WNorm);
    cudaFree(velocity.BNorm);
    cudaFree(velocity.qkvw);
    cudaFree(velocity.qkvb);
    cudaFree(velocity.projW);
    cudaFree(velocity.projB);
    cudaFree(velocity.moegw1);
    cudaFree(velocity.moegb1);
    cudaFree(velocity.moew1);
    cudaFree(velocity.moeb1);
    cudaFree(velocity.moegw2);
    cudaFree(velocity.moegb2);
    cudaFree(velocity.moew2);
    cudaFree(velocity.moeb2);
    cudaFree(velocity.WCLS);
    cudaFree(velocity.BCLS);

    cudaFree(momentum.CLS);
    cudaFree(momentum.WNorm);
    cudaFree(momentum.BNorm);
    cudaFree(momentum.qkvw);
    cudaFree(momentum.qkvb);
    cudaFree(momentum.projW);
    cudaFree(momentum.projB);
    cudaFree(momentum.moegw1);
    cudaFree(momentum.moegb1);
    cudaFree(momentum.moew1);
    cudaFree(momentum.moeb1);
    cudaFree(momentum.moegw2);
    cudaFree(momentum.moegb2);
    cudaFree(momentum.moew2);
    cudaFree(momentum.moeb2);
    cudaFree(momentum.WCLS);
    cudaFree(momentum.BCLS);

    cudaFree(buffers.theta);
    cudaFree(buffers.attnNormX);
    cudaFree(buffers.attnNorm);
    cudaFree(buffers.RMS);
    cudaFree(buffers.qkv);
    cudaFree(buffers.Q);
    cudaFree(buffers.K);
    cudaFree(buffers.V);
    cudaFree(buffers.QR);
    cudaFree(buffers.KR);
    cudaFree(buffers.KT);
    cudaFree(buffers.attnScores);
    cudaFree(buffers.attnWeights);
    cudaFree(buffers.attnY);
    cudaFree(buffers.projX);
    cudaFree(buffers.projY);
    cudaFree(buffers.moeGate1);
    cudaFree(buffers.moeProbs1);
    cudaFree(buffers.moeSorted1);
    cudaFree(buffers.moeIndices1);
    cudaFree(buffers.moeW1);
    cudaFree(buffers.moeB1);
    cudaFree(buffers.moeExperts1);
    cudaFree(buffers.moeY1);
    cudaFree(buffers.moeHidden);
    cudaFree(buffers.moeGate2);
    cudaFree(buffers.moeProbs2);
    cudaFree(buffers.moeSorted2);
    cudaFree(buffers.moeIndices2);
    cudaFree(buffers.moeW2);
    cudaFree(buffers.moeB2);
    cudaFree(buffers.moeExperts2);
    cudaFree(buffers.moeY2);
    cudaFree(buffers.decoderY);
    cudaFree(buffers.CLSNorm);
    cudaFree(buffers.ZNormCLS);
    cudaFree(buffers.ZCLS);

    cudaFree(backbuffers.dL_dXNorm);
    cudaFree(backbuffers.dL_dXCLS);
    cudaFree(backbuffers.loadBalance2);
    cudaFree(backbuffers.dL_dMoeY2);
    cudaFree(backbuffers.Xtmp2);
    cudaFree(backbuffers.dL_dWtmp2);
    cudaFree(backbuffers.dL_dBtmp2);
    cudaFree(backbuffers.dE_dX2);
    cudaFree(backbuffers.Wtmp2);
    cudaFree(backbuffers.Etmp2);
    cudaFree(backbuffers.EMA2);
    cudaFree(backbuffers.J2);
    cudaFree(backbuffers.dL_dG2);
    cudaFree(backbuffers.dG_dX2);
    cudaFree(backbuffers.dG_dZ2);
    cudaFree(backbuffers.dL_dXtmp2);
    cudaFree(backbuffers.dL_dXsum2);
    cudaFree(backbuffers.dL_dMoeHidden);
    cudaFree(backbuffers.loadBalance1);
    cudaFree(backbuffers.dL_dMoeY1);
    cudaFree(backbuffers.Xtmp1);
    cudaFree(backbuffers.dL_dWtmp1);
    cudaFree(backbuffers.dL_dBtmp1);
    cudaFree(backbuffers.dE_dX1);
    cudaFree(backbuffers.Wtmp1);
    cudaFree(backbuffers.Etmp1);
    cudaFree(backbuffers.EMA1);
    cudaFree(backbuffers.J1);
    cudaFree(backbuffers.dL_dG1);
    cudaFree(backbuffers.dG_dX1);
    cudaFree(backbuffers.dG_dZ1);
    cudaFree(backbuffers.dL_dXtmp1);
    cudaFree(backbuffers.dL_dXsum1);
    cudaFree(backbuffers.dL_dProjX);
    cudaFree(backbuffers.dL_dY);
    cudaFree(backbuffers.dL_dW);
    cudaFree(backbuffers.dL_dK);
    cudaFree(backbuffers.dL_dQ);
    cudaFree(backbuffers.dL_dV);
    cudaFree(backbuffers.J);
    cudaFree(backbuffers.dL_dScores);
    cudaFree(backbuffers.dL_dKT);
    cudaFree(backbuffers.dL_dQR);
    cudaFree(backbuffers.dL_dKR);
    cudaFree(backbuffers.dL_dqkv);
    cudaFree(backbuffers.dL_dXtmp);
}



