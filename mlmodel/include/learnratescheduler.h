#ifndef LEARNRATESCHEDULER_H
#define LEARNRATESCHEDULER_H
#include <cmath>

float calculateLearnRate (float learnRate, float lrMin, int warmupSteps, int currentStep, int totalSteps) {
    if (currentStep < warmupSteps) {
        return learnRate * (static_cast<float>(currentStep) / static_cast<float>(warmupSteps));
    } else {
        return lrMin +  (learnRate - lrMin) * ((1.0f + cosf(M_PI * static_cast<float>(currentStep - warmupSteps + 1) / static_cast<float>(totalSteps - warmupSteps)))) / (1 + cosf(M_PI * static_cast<float>(currentStep - warmupSteps) / static_cast<float>(totalSteps - warmupSteps)));
    }
}


#endif