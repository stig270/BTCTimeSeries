#ifndef CONFIG_H
#define CONFIG_H

#include <vector>


struct ModelHyperparameters {
    const int T = 32; // Sequence length
    const int C = 32; // Number of channels
    const int N = 4; // Number of heads (attention)
    const int H = C/N; // Head size (8)
    const int E = 4; // Number of experts (MoE)
    const int L = 4; // Number of layers
    const int F = 2; // Number of classes in the classifier
    const int K = 2; // Top-K experts to use in MoE
    const int B = 10000; // Base parameter for positional encoding
    const float epsilon = 1e-5f; // Small constant for numerical stability in division operations
};

struct ModelParameters{
    float* CLS; //CLS token, [C]
    float* WNorm; // [2L + 1, C, C]
    float* BNorm; // [2L + 1, C]
    float* qkvw; // [L, C, 3*N*H]
    float* qkvb; // [L, 3*N*H]
    float* projW; // [L, C, C]
    float* projB; // [L, C]
    float* moegw1; // [L, C, E]
    float* moegb1; // [L, E]
    float* moew1; // [L, E, C, C*4]
    float* moeb1; // [L, E, C*4]
    float* moegw2; // [L, C*4, E]
    float* moegb2; // [L, E]
    float* moew2; // [L, E, C*4, C]
    float* moeb2; // [L, E, C]
    float* WCLS; // [C, F]
    float* BCLS; // [F]
};

struct HostParameters{
    std::vector<float> CLS; //CLS token, [C]
    std::vector<float> WNorm; // [2L + 1, C, C]
    std::vector<float> BNorm; // [2L + 1, C]
    std::vector<float> qkvw; // [L, C, 3*N*H]
    std::vector<float> qkvb; // [L, 3*N*H]
    std::vector<float> projW; // [L, C, C]
    std::vector<float> projB; // [L, C]
    std::vector<float> moegw1; // [L, C, E]
    std::vector<float> moegb1; // [L, E]
    std::vector<float> moew1; // [L, E, C, C*4]
    std::vector<float> moeb1; // [L, E, C*4]
    std::vector<float> moegw2; // [L, C*4, E]
    std::vector<float> moegb2; // [L, E]
    std::vector<float> moew2; // [L, E, C*4, C]
    std::vector<float> moeb2; // [L, E, C]
    std::vector<float> WCLS; // [C, F]
    std::vector<float> BCLS; // [F]
};

struct ParameterSizes{
    size_t CLS; //CLS token, [C]
    size_t WNorm; // [2L + 1, C, C]
    size_t BNorm; // [2L + 1, C]
    size_t qkvw; // [L, C, 3*N*H]
    size_t qkvb; // [L, 3*N*H]
    size_t projW; // [L, C, C]
    size_t projB; // [L, C]
    size_t moegw1; // [L, C, E]
    size_t moegb1; // [L, E]
    size_t moew1; // [L, E, C, C*4]
    size_t moeb1; // [L, E, C*4]
    size_t moegw2; // [L, C*4, E]
    size_t moegb2; // [L, E]
    size_t moew2; // [L, E, C*4, C]
    size_t moeb2; // [L, E, C]
    size_t WCLS; // [C, F]
    size_t BCLS; // [F]
};

struct TrainingConfig {
    float weightDecay = 1e-5f; // Weight decay for regularization
    float alpha = 0.001f; // Moe Load balance loss alpha
    float beta = 0.999f; // Moe EMA beta
    float beta1 = 0.9f; // Adam optimizer beta1
    float beta2 = 0.999f; // Adam optimizer beta2
    float initialLR = 1e-3f; // Initial learning rate
    float lrMin = 1e-5f; // Minimum learning rate
    int warmupSteps = 100; // Learning rate warmup steps
    int epochs = 100; // Number of training epochs
    int saveEvery = 2000; // Save model every n steps
    int logEvery = 500; // Log training metrics every n steps
};

struct IntermediateBuffers {
    float* theta;
    float* RMS;
    float* attnNormX;
    float* attnNorm;
    float* attnX;
    float* qkv;
    float* Q;
    float* K;
    float* V;
    float* QR;
    float* KR;
    float* KT;
    float* attnScores;
    float* attnWeights;
    float* attnY;
    float* projX;
    float* projY;
    float* moeNormX;
    float* moeNorm;
    float* moeX;
    float* moeGate1;
    float* moeProbs1;
    float* moeSorted1;
    int* moeIndices1; // do not += (T-1)*E here
    float* moeW1;
    float* moeB1;
    float* moeExperts1;
    float* moeY1;
    float* moeHidden;
    float* moeGate2;
    float* moeProbs2;
    float* moeSorted2;
    int* moeIndices2; // do not += (T-1)*E here
    float* moeW2;
    float* moeB2;
    float* moeExperts2;
    float* moeY2;
    float* decoderY;
    float* CLSNorm;
    float* ZNormCLS;
    float* ZCLS;
};

struct BackpropagationBuffers{
    float* dL_dXNorm;
    float* dL_dXCLS;
    float* loadBalance2;
    float* dL_dMoeY2;
    float* Xtmp2;
    float* dL_dWtmp2;
    float* dL_dBtmp2;
    float* dE_dX2;
    float* Wtmp2;
    float* Etmp2;
    float* EMA2; // Do not zero after step
    float* J2;
    float* dL_dG2;
    float* dG_dX2;
    float* dG_dZ2;
    float* dL_dXtmp2;
    float* dL_dXsum2;
    float* dL_dMoeHidden;
    float* loadBalance1;
    float* dL_dMoeY1;
    float* Xtmp1;
    float* dL_dWtmp1;
    float* dL_dBtmp1;
    float* dE_dX1;
    float* Wtmp1;
    float* Etmp1;
    float* EMA1; // Do not zero after step
    float* J1;
    float* dL_dG1;
    float* dG_dX1;
    float* dG_dZ1;
    float* dL_dXtmp1;
    float* dL_dXsum1;
    float* dL_dProjX;
    float* dL_dY;
    float* dL_dW;
    float* dL_dV;
    float* dL_dK;
    float* dL_dQ;
    float* J;
    float* dL_dScores;
    float* dL_dKT;
    float* dL_dKR;
    float* dL_dQR;
    float* dL_dqkv;
    float* dL_dXtmp;
};

struct Parameter{
    float* parameter;
    float* gradient;
    float* velocity;
    float* momentum;
};
#endif