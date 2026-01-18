#ifndef LEARNRATESCHEDULER_H
#define LEARNRATESCHEDULER_H
#include <cmath>

float calculateLearnRate (float learnRate, float lrMin, int warmupSteps, int currentStep, int totalSteps, float epsilon = 1e-8f) {
    if (currentStep < warmupSteps) {
        return learnRate * (static_cast<float>(currentStep) / static_cast<float>(warmupSteps)) + epsilon;
    } else {
        return lrMin +  (learnRate - lrMin) * 0.5f * (1.0f + cosf(M_PI * (static_cast<float>(currentStep - warmupSteps) / static_cast<float>(totalSteps - warmupSteps))));
    }
}


#endif