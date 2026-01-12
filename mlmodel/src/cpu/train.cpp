#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include "transformer.cu"
#include "config.h"
#include "learnratesheduler.h"
#include "model.h"

void runTrainingLoop(const ModelHyperparameters& config, const TrainingConfig& trainConfig, const char* modelPath){
    
    const float weightDecay = trainConfig.weightDecay;
    const float beta1 = trainConfig.beta1;
    const float beta2 = trainConfig.beta2;
    const float initialLR = trainConfig.initialLR;
    const float lrMin = trainConfig.lrMin;
    const int warmupSteps = trainConfig.warmupSteps;
    const int epochs = trainConfig.epochs;
    const int saveEvery = trainConfig.saveEvery;
    const int evaluateEvery = trainConfig.evaluateEvery;

    //load data

    // load or initialize parameters

    float* X;
    float* Y;
    float* expectedOutput;
    float* loss;

    size_t XSize = config.T * config.C * sizeof(float); 
    size_t YSize = config.F * sizeof(float);
    size_t expectedOutputSize = config.F * sizeof(float);
    size_t lossSize = sizeof(float);

    cudaMalloc(&X, XSize);
    cudaMalloc(&Y, YSize);
    cudaMalloc(&expectedOutput, expectedOutputSize);
    cudaMalloc(&loss, lossSize);

    std::vector<float> XHost(XSize);
    std::vector<float> expectedOutputHost(expectedOutputSize);
    float lossHost; 

    // init optimizer, lr scheduler

    float learnRate = initialLR;
    auto trainingStart = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < epochs; t++) {
       
        auto epochStart = std::chrono::high_resolution_clock::now();

        learnRate = calculateLearnRate(initialLR, lrMin,  warmupSteps, t, epochs);
        // iterate over training examples for this
        runTransformerForward(X, Y, config, /*params*/);

        computeLoss(Y, expectedOutput, loss, config.F);

        runTransformerBackward(X, Y, expectedOutput, config, /*params*/);


        

        if (t % logEvery == 0) {
            cudaMemcpy(&lossHost, loss, lossSize, cudaMemcpyDeviceToHost);
            std::cout << "Step " << t << ", Loss: " << lossHost << std::endl;
        }

        if (t % saveEvery == 0) {
            // save model parameters
        }

        if (t % evaluateEvery == 0) {
            // evaluate on validation set
        }

        t++;
    }   







}

