#include <fstream>
#include <cstring>
#include <random>
#include <cmath>
#include "config.h"
#include "model.h"

void saveParams(const ParameterSizes& parameterSizes, const HostParameters& params, const ModelConfig& config, const std::string& path){
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(params.CLS.data()), parameterSizes.CLS * sizeof(float));
    file.write(reinterpret_cast<const char*>(params.WNorm.data()), parameterSizes.WNorm * sizeof(float));
    file.write(reinterpret_cast<const char*>(params.BNorm.data()), parameterSizes.BNorm * sizeof(float));
    file.write(reinterpret_cast<const char*>(params.qkvw.data()), parameterSizes.qkvw * sizeof(float));
    file.write(reinterpret_cast<const char*>(params.qkvb.data()), parameterSizes.qkvb * sizeof(float));
    file.write(reinterpret_cast<const char*>(params.projW.data()), parameterSizes.projW * sizeof(float));
    file.write(reinterpret_cast<const char*>(params.projB.data()), parameterSizes.projB * sizeof(float));
    file.write(reinterpret_cast<const char*>(params.moegw1.data()), parameterSizes.moegw1 * sizeof(float));
    file.write(reinterpret_cast<const char*>(params.moegb1.data()), parameterSizes.moegb1 * sizeof(float));
    file.write(reinterpret_cast<const char*>(params.moew1.data()), parameterSizes.moew1 * sizeof(float));
    file.write(reinterpret_cast<const char*>(params.moeb1.data()), parameterSizes.moeb1 * sizeof(float));
    file.write(reinterpret_cast<const char*>(params.moegw2.data()), parameterSizes.moegw2 * sizeof(float));
    file.write(reinterpret_cast<const char*>(params.moegb2.data()), parameterSizes.moegb2 * sizeof(float));
    file.write(reinterpret_cast<const char*>(params.moew2.data()), parameterSizes.moew2 * sizeof(float));
    file.write(reinterpret_cast<const char*>(params.moeb2.data()), parameterSizes.moeb2 * sizeof(float));
    file.write(reinterpret_cast<const char*>(params.WCLS.data()), parameterSizes.WCLS * sizeof(float));
    file.write(reinterpret_cast<const char*>(params.BCLS.data()), parameterSizes.BCLS * sizeof(float));
    file.close();
}

HostParameters loadParams(const ParameterSizes& parameterSizes, const ModelConfig& config, const std::string& path){
    HostParameters hostParams = new HostParameters;

    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(hostParams.CLS.data()), parameterSizes.CLS * sizeof(float));
    file.read(reinterpret_cast<char*>(hostParams.WNorm.data()), parameterSizes.WNorm * sizeof(float));
    file.read(reinterpret_cast<char*>(hostParams.BNorm.data()), parameterSizes.BNorm * sizeof(float));
    file.read(reinterpret_cast<char*>(hostParams.qkvw.data()), parameterSizes.qkvw * sizeof(float));
    file.read(reinterpret_cast<char*>(hostParams.qkvb.data()), parameterSizes.qkvb * sizeof(float));
    file.read(reinterpret_cast<char*>(hostParams.projW.data()), parameterSizes.projW * sizeof(float));
    file.read(reinterpret_cast<char*>(hostParams.projB.data()), parameterSizes.projB * sizeof(float));
    file.read(reinterpret_cast<char*>(hostParams.moegw1.data()), parameterSizes.moegw1 * sizeof(float));
    file.read(reinterpret_cast<char*>(hostParams.moegb1.data()), parameterSizes.moegb1 * sizeof(float));
    file.read(reinterpret_cast<char*>(hostParams.moew1.data()), parameterSizes.moew1 * sizeof(float));
    file.read(reinterpret_cast<char*>(hostParams.moeb1.data()), parameterSizes.moeb1 * sizeof(float));
    file.read(reinterpret_cast<char*>(hostParams.moegw2.data()), parameterSizes.moegw2 * sizeof(float));
    file.read(reinterpret_cast<char*>(hostParams.moegb2.data()), parameterSizes.moegb2 * sizeof(float));
    file.read(reinterpret_cast<char*>(hostParams.moew2.data()), parameterSizes.moew2 * sizeof(float));
    file.read(reinterpret_cast<char*>(hostParams.moeb2.data()), parameterSizes.moeb2 * sizeof(float));
    file.read(reinterpret_cast<char*>(hostParams.WCLS.data()), parameterSizes.WCLS * sizeof(float));
    file.read(reinterpret_cast<char*>(hostParams.BCLS.data()), parameterSizes.BCLS * sizeof(float));
    file.close();

    return hostParams;
}

HostParameters initializeParams(const ParameterSizes& parameterSizes, const ModelConfig& config){

    HostParameters hostParams = new HostParameters;

    const int C = config.C;
    const int L = config.L;
    const int H = config.H;
    const int N = config.N;
    const int E = config.E;
    const int F = config.F;

    std::random_device rd;
    std::mt19937 gen(rd());

    auto xavierInit = [&](std::vector<float>& array, size_t size, int fanIn, int fanOut){
        float limit = sqrtf(6.0f / (fanIn + fanOut));
        std::uniform_real_distribution<float> dis(-limit, limit);
        for(int i = 0; i < size; i++){
            array[i] = dis(gen);
        }
    };

    auto zeroInit = [&](std::vector<float>& array, size_t size){
        std::memset(array.data(), 0, size * sizeof(float));
    };

    auto randomInit = [&](std::vector<float>& array, size_t size){
        std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
        for(int i = 0; i < size; i++){
            array[i] = dis(gen);
        }
    };

    hostParams.CLS = std::vector<float>(parameterSizes.CLS);
    randomInit(hostParams.CLS, parameterSizes.CLS);

    hostParams.WNorm = std::vector<float>(parameterSizes.WNorm);
    xavierInit(hostParams.WNorm, parameterSizes.WNorm, C, C);
    hostParams.BNorm = std::vector<float>(parameterSizes.BNorm);
    zeroInit(hostParams.BNorm, parameterSizes.BNorm);

    hostParams.qkvw = std::vector<float>(parameterSizes.qkvw);
    xavierInit(hostParams.qkvw, parameterSizes.qkvw, C, 3 * N * H);

    hostParams.qkvb = std::vector<float>(parameterSizes.qkvb);
    zeroInit(hostParams.qkvb, parameterSizes.qkvb);

    hostParams.projW = std::vector<float>(parameterSizes.projW);
    xavierInit(hostParams.projW, parameterSizes.projW, C, C);

    hostParams.projB = std::vector<float>(parameterSizes.projB);
    zeroInit(hostParams.projB, parameterSizes.projB);

    hostParams.moegw1 = std::vector<float>(parameterSizes.moegw1);
    xavierInit(hostParams.moegw1, parameterSizes.moegw1, C, E);

    hostParams.moegb1 = std::vector<float>(parameterSizes.moegb1);
    zeroInit(hostParams.moegb1, parameterSizes.moegb1);

    hostParams.moew1 = std::vector<float>(parameterSizes.moew1);
    xavierInit(hostParams.moew1, parameterSizes.moew1, C, C * 4);

    hostParams.moeb1 = std::vector<float>(parameterSizes.moeb1);
    zeroInit(hostParams.moeb1, parameterSizes.moeb1);

    hostParams.moegw2 = std::vector<float>(parameterSizes.moegw2);
    xavierInit(hostParams.moegw2, parameterSizes.moegw2, C * 4, E);

    hostParams.moegb2 = std::vector<float>(parameterSizes.moegb2);
    zeroInit(hostParams.moegb2, parameterSizes.moegb2);

    hostParams.moew2 = std::vector<float>(parameterSizes.moew2);
    xavierInit(hostParams.moew2, parameterSizes.moew2, C * 4, C);

    hostParams.moeb2 = std::vector<float>(parameterSizes.moeb2);
    zeroInit(hostParams.moeb2, parameterSizes.moeb2);

    hostParams.WCLS = std::vector<float>(parameterSizes.WCLS);
    xavierInit(hostParams.WCLS, parameterSizes.WCLS, C, F);

    hostParams.BCLS = std::vector<float>(parameterSizes.BCLS);
    zeroInit(hostParams.BCLS, parameterSizes.BCLS);
    

    return hostParams;
}