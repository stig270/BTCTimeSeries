#include <iostream>
#include <vector>
#include <fstream>
#include "config.h"
#include "model.h"
#include "transformer.cu"

void runInference(const ModelConfig& config, const std::string& modelPath, const std::string& inputPath){

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

    ParameterSizes parameterSizes = new ParameterSizes;
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

    ModelParameters params = new ModelParameters;
    IntermediateBuffers buffers = new IntermediateBuffers;  

    HostParameters hostParameters = loadParams(parameterSizes, config, modelPath);

    std::string line;
    std::vector<std::vector<float>> test_data;
    std::ifstream file(inputPath);

    std::getline(file, line);
    
    while (std::getline(file, line) && test_data.size() < T - 1) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> features;
        
        std::getline(ss, cell, ',');
        
        std::vector<std::string> cells;
        while (std::getline(ss, cell, ',')) {
            cells.push_back(cell);
        }
        
        for (size_t i = 0; i < cells.size() - 1; ++i) {
            features.push_back(std::stof(cells[i]));
        }
      
        
        test_data.push_back(features);
    }
    
    file.close();

    float* X; 
    float* Y; 

    size_t XSize = T * C * sizeof(float); 
    size_t YSize = F * sizeof(float);

    cudaMalloc(&X, XSize);
    cudaMalloc(&Y, YSize);

    std::vector<float> XHost(XSize);
    std::vector<float> YHost(YSize);

    for(int t = 0; t < T - 1; t++){
        for(int c = 0; c < C; c++){
            XHost[t * C + c] = test_data[t][c];
        }
    }
    
    params.CLS = X + (T - 1) * C;
    cudaMalloc(&params.CLS, parameterSizes.CLS * sizeof(float));
    cudaMemcpy(params.CLS, hostParameters.CLS.data(), parameterSizes.CLS * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&params.WNorm, parameterSizes.WNorm * sizeof(float));
    cudaMemcpy(params.WNorm, hostParameters.WNorm.data(), parameterSizes.WNorm * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&params.BNorm, parameterSizes.BNorm * sizeof(float));
    cudaMemcpy(params.BNorm, hostParameters.BNorm.data(), parameterSizes.BNorm * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&params.qkvw, parameterSizes.qkvw * sizeof(float));
    cudaMemcpy(params.qkvw, hostParameters.qkvw.data(), parameterSizes.qkvw * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&params.qkvb, parameterSizes.qkvb * sizeof(float));
    cudaMemcpy(params.qkvb, hostParameters.qkvb.data(), parameterSizes.qkvb * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&params.projW, parameterSizes.projW * sizeof(float));
    cudaMemcpy(params.projW, hostParameters.projW.data(), parameterSizes.projW * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&params.projB, parameterSizes.projB * sizeof(float));
    cudaMemcpy(params.projB, hostParameters.projB.data(), parameterSizes.projB * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&params.moegw1, parameterSizes.moegw1 * sizeof(float));
    cudaMemcpy(params.moegw1, hostParameters.moegw1.data(), parameterSizes.moegw1 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&params.moegb1, parameterSizes.moegb1 * sizeof(float));
    cudaMemcpy(params.moegb1, hostParameters.moegb1.data(), parameterSizes.moegb1 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&params.moew1, parameterSizes.moew1 * sizeof(float));
    cudaMemcpy(params.moew1, hostParameters.moew1.data(), parameterSizes.moew1 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&params.moeb1, parameterSizes.moeb1 * sizeof(float));
    cudaMemcpy(params.moeb1, hostParameters.moeb1.data(), parameterSizes.moeb1 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&params.moegw2, parameterSizes.moegw2 * sizeof(float));
    cudaMemcpy(params.moegw2, hostParameters.moegw2.data(), parameterSizes.moegw2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&params.moegb2, parameterSizes.moegb2 * sizeof(float));
    cudaMemcpy(params.moegb2, hostParameters.moegb2.data(), parameterSizes.moegb2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&params.moew2, parameterSizes.moew2 * sizeof(float));
    cudaMemcpy(params.moew2, hostParameters.moew2.data(), parameterSizes.moew2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&params.moeb2, parameterSizes.moeb2 * sizeof(float));
    cudaMemcpy(params.moeb2, hostParameters.moeb2.data(), parameterSizes.moeb2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&params.WCLS, parameterSizes.WCLS * sizeof(float));
    cudaMemcpy(params.WCLS, hostParameters.WCLS.data(), parameterSizes.WCLS * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&params.BCLS, parameterSizes.BCLS * sizeof(float));
    cudaMemcpy(params.BCLS, hostParameters.BCLS.data(), parameterSizes.BCLS * sizeof(float), cudaMemcpyHostToDevice);

    
    cudaMalloc(&buffers.theta, T * C * sizeof(float));
    cudaMalloc(&buffers.attnNormX, L * T * C * sizeof(float));
    cudaMalloc(&buffers.attnNorm, L * T * C * sizeof(float));
    cudaMalloc(&buffers.attnRMS, L * sizeof(float));
    cudaMalloc(&buffers.qkv, L * 3 * N * H * T * sizeof(float));
    cudaMalloc(&buffers.Q, L * N * H * T * sizeof(float));
    cudaMalloc(&buffers.K, L * N * H * T * sizeof(float));
    cudaMalloc(&buffers.V, L * N * H * T * sizeof(float));
    cudaMalloc(&buffers.QR, L * N * H * T * sizeof(float));
    cudaMalloc(&buffers.KR, L * N * H * T * sizeof(float));
    cudaMalloc(&buffers.KT, L * N * H * T * sizeof(float));
    cudaMalloc(&buffers.attnScores, L * N * H * T * T * sizeof(float));
    cudaMalloc(&buffers.attnWeights, L * N * H * T * T * sizeof(float));
    cudaMalloc(&buffers.attnY, L * T * C * sizeof(float));
    cudaMalloc(&buffers.projX, L * T * C * sizeof(float));
    cudaMalloc(&buffers.projY, L * T * C * sizeof(float));
    cudaMalloc(&buffers.moeGate1, L * T * E * sizeof(float));
    cudaMalloc(&buffers.moeProbs1, L * T * E * sizeof(float));
    cudaMalloc(&buffers.moeSorted1, L * T * E * sizeof(float));
    cudaMalloc(&buffers.moeIndices1, L * T * E * sizeof(int));
    cudaMalloc(&buffers.moeW1, L * K * C * C * 4 *sizeof(float));
    cudaMalloc(&buffers.moeB1, L * K * C * 4 * sizeof(float));
    cudaMalloc(&buffers.moeExperts1, L * K * C * 4  * sizeof(float));
    cudaMalloc(&buffers.moeY1, L * T * C * 4 * sizeof(float));
    cudaMalloc(&buffers.moeHidden1, L * T * C * 4 * sizeof(float));
    cudaMalloc(&buffers.moeGate2, L * T * E * sizeof(float));
    cudaMalloc(&buffers.moeProbs2, L * T * E * sizeof(float));
    cudaMalloc(&buffers.moeSorted2, L * T * E * sizeof(float));
    cudaMalloc(&buffers.moeIndices2, L * T * E * sizeof(int));
    cudaMalloc(&buffers.moeW2, L * K * C * 4 * C * sizeof(float));
    cudaMalloc(&buffers.moeB2, L * K * C * sizeof(float));
    cudaMalloc(&buffers.moeExperts2, L * K * C * sizeof(float));
    cudaMalloc(&buffers.moeY2, L * T * C * sizeof(float));
    cudaMalloc(&buffers.decoderY, T * C * sizeof(float));
    cudaMalloc(&buffers.CLSNorm, C * sizeof(float));
    cudaMalloc(&buffers.ZNormCLS, C * sizeof(float));
    cudaMalloc(&buffers.ZCLS, F * sizeof(float)); 

    calculateThetaLauncher(buffers.theta, T, H, B);

    runTransformerForward(X, Y, params, config, buffers);

    cudaMemcpy(YHost.data(), Y, YSize, cudaMemcpyDeviceToHost);

    std::cout << "Price Increase: " << YHost[0] * 100 << "%" << std::endl;
    std::cout << "Price Decrease: " << YHost[1] * 100 << "%" << std::endl;
    

    cudaFree(X);
    cudaFree(Y);
    cudaFree(params.CLS);
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

    cudaFree(buffers.theta);
    cudaFree(buffers.attnNormX);
    cudaFree(buffers.attnNorm);
    cudaFree(buffers.attnRMS);
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
    cudaFree(buffers.moeHidden1);
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

    
}