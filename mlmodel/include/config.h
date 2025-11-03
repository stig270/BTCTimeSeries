#ifndef CONFIG_H
#define CONFIG_H

struct ModelHyperparameters {
    int T = 32; // Sequence length
    int C = 32; // Number of channels
    int N = 4; // Number of heads (attention)
    int H = C/N; // Head size
    int E = 4; // Number of experts (MoE)
    int L = 4; // Number of layers
    int F = 2; // Number of classes in the classifier
    int K = 2; // Top-K experts to use in MoE
    float epsilon = 1e-10; // Small constant for numerical stability in division operations
};

struct ModelParameters{
    float* CLS; //CLS token, [C]
    float* WNorm; // [2L + 1, C, C]
    float* BNorm; // [2L + 1, C]
    float* qkvw; // [L, C, 3*N*H]
    float* qkvb; // [L, 3*N*H]
    float* proj; // [L, C, C]
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
}


struct TrainingConfig {
    float weightDecay = 1e-5f; // Weight decay for regularization
    float beta1 = 0.9f; // Adam optimizer beta1
    float beta2 = 0.999f; // Adam optimizer beta2
    float initialLR = 1e-3f; // Initial learning rate
    int warmupSteps = 100; // Learning rate warmup steps
    int epochs = 100; // Number of training epochs
    int saveEvery = 200; // Save model every n steps
    int evaluateEvery = 100; // Evaluate model every n steps
    int logEvery = 10; // Log training metrics every n steps
};

struct IntermediateBuffers {
    float* attnNormX;
    float* attnNorm;
    float* attnX;
    float* qkv;
    float* Q;
    float* K;
    float* V;
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
    float* moeIndices1;
    float* moeExperts1;
    float* moeOut1;
    float* moeHidden;
    float* moeGate2;
    float* moeProbs2;
    float* moeSorted2;
    float* moeIndices2;
    float* moeExperts2;
    float* moeOut2;
    float* CLSNorm;
    float* ZNormCLS;
    float* ZCLS;



};

#endif