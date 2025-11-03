#include <iostream>
#include <string>
#include <cstring>
#include <cuda_runtime.h>
#include "config.h"
#include "model.h"

int main(int argc, char** argv) {
    std::string mode = "infer"; //default is inference
    std::string modelPath = "model.json";
    
    for(uint i = 1; i < argc; i++){
        if(strcmp(argv[i], "train") == 0){
            mode = "train";
        }
    }

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    ModelHyperparameters config = new ModelHyperparameters;

    if(mode == "train"){
        runTrainingLoop(config);
    }
    else{
        runInference(config)
    }


    return 0;
} 

