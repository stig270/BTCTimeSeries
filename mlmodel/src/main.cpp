#include <iostream>
#include <string>
#include <cstring>
#include <cuda_runtime.h>
#include "config.h"
#include "model.h"
#include "dataprocessing.h"

void checkCudaError(const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << "Error code: " << err << std::endl;
        std::cerr << "Error string: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    } else {
        std::cout << "No CUDA error at " << file << ":" << line << std::endl;
    }
}

int main(int argc, char** argv) {
    std::string mode = "infer"; //default is inference
    bool initParameters = false;
    for(int i = 1; i < argc; i++){
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

    const ModelHyperparameters config;
    const TrainingConfig trainConfig;


    if(mode == "train"){
        TimeSeriesDataset data("data.csv");

        DatasetSplitter splitter(data, 0.8f, config.T);

        runTrainingLoop(config, trainConfig, "model.bin", data, splitter.train_timestamps, splitter.val_timestamps, initParameters);
    }
    else{
        runInference(config, "model.bin", "input.csv");
    }


    return 0;
} 

