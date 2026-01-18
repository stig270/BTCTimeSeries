#include <iostream>
#include <string>
#include <cstring>
#include <cuda_runtime.h>
#include "config.h"
#include "model.h"
#include "dataprocessing.h"


int main(int argc, char** argv) {
    std::string mode = "infer"; //default is inference
    std::string modelPath = "model.bin";
    bool initParameters = false;

    for(uint i = 1; i < argc; i++){
        if(strcmp(argv[i], "train") == 0){
            mode = "train";
        }
        if(strcmp(argv[i], "initparams") == 0){
            initParameters = true;
        }
    }

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    ModelHyperparameters config = new ModelHyperparameters;
    TrainingConfig trainConfig = new TrainingConfig;


    if(mode == "train"){
        TimeSeriesDataset data("data.csv");

        DatasetSplitter splitter(data, 0.8f, config.T);

        runTrainingLoop(config, trainConfig, modelPath.c_str(), data, splitter.train_timestamps, splitter.val_timestamps, initParameters);
    }
    else{
        runInference(config, modelPath.c_str(), "input.csv");
    }


    return 0;
} 

