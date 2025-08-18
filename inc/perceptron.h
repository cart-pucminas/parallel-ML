#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "dataset.h"
#include "sgd.h"

typedef struct
{
    int hiddenLayerCount;
    int *hiddenLayerSizes;
    float learningRate;
    ActivationFunction activation;
    int miniBatchSize;
    int epochs;
} Params;

void setActivationFunction(ActivationFunction af);
void learn(Dataset *dataset);
void classify(Dataset *dataset);

#endif
