#ifndef MODEL_H
#define MODEL_H

#include "dataprocessing.h"
#include "config.h"
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

void runTrainingLoop(const ModelHyperparameters& config, const TrainingConfig& trainConfig, const std::string& modelPath,
    TimeSeriesDataset& data, const std::vector<std::string>& train_timestamps, const std::vector<std::string>& val_timestamps,
     const bool initParameters);
void runInference(const ModelHyperparameters& config, const std::string& modelPath, const std::string& inputPath);
void saveParams(const ParameterSizes& parameterSizes, const HostParameters& params, const std::string& path);
HostParameters loadParams(const ParameterSizes& parameterSizes, const std::string& path);
HostParameters initializeParams(const ParameterSizes& parameterSizes, const ModelHyperparameters& config);
void calculateThetaLauncher(float* theta, int T, int H, int B);
void calculateLoss(float* Y, float* P, float* loss, int F, float epsilon);
void transformerForward(float* X, float* Y, ModelParameters& params, const ModelHyperparameters& config, IntermediateBuffers& buffers);
void transformerBackward(ModelParameters& params, ModelParameters& gradients, ModelParameters& velocity, ModelParameters& momentum,
     const ModelHyperparameters& config, float learningRate, float alpha, float beta, float beta1, float beta2, float l2Lambda, int time,
      IntermediateBuffers& buffers, BackpropagationBuffers& backbuffers, float* Y, float* P);
void checkCudaError(const char* file, int line);

#endif