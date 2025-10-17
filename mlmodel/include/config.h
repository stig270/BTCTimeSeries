#ifndef CONFIG_H
#define CONFIG_H

struct ModelHyperparameters {
    int T = 32; // Sequence length
    int C = 32; // Number of channels
    int N = 4; // Number of heads (attention)
    int H = C/N; // Head size
    int E = 4; // Number of experts (MoE)
    int L = 4; // Number of layers
    int epsilon = 1e-10; // Small constant for numerical stability in division operations

    
};

struct TrainingConfig {
    int B = 16; // Batch size
    float weight_decay = 1e-5f; // Weight decay for regularization
    float beta1 = 0.9f; // Adam optimizer beta1
    float beta2 = 0.999f; // Adam optimizer beta2
    float initial_lr = 1e-3f; // Initial learning rate
    int warmup_steps = 100; // Learning rate warmup steps
    int epochs = 32; // Number of training epochs
};
#endif