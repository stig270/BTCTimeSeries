#ifndef MODEL_H
#define MODEL_H

#include "config.h"
#include <vector>

void runTrainingLoop(const ModelHyperparameters& config, const TrainingConfig& trainConfig, const char* modelPath);
void runInference(const ModelParameters& params, const ModelConfig& config);
void saveParams(const ModelParameters& params, const ModelConfig& config, const char* path)
ModelParameters loadParams(ModelParameters& params, const ModelConfig& config, const char* path);
ModelParameters initializeParams(ModelParameters& params, const ModelConfig& config);

#endif