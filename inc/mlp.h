#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <stdlib.h>

typedef struct
{
    size_t epochs;
    float learningRate;
    size_t miniBatchSize;
    float **weights;
    float **biases;
    size_t *layersSizes;
    size_t layerCount;
    size_t maxLayerSize;
    size_t totalNeurons;
    size_t totalSynapses;
} Network;

typedef struct
{
    float **inputs;
    float **groundTruths;
    size_t size;
} Dataset;

Network *constructNetwork(size_t epochs, size_t layerCount, size_t *layersSizes,
                          float learningRate, size_t miniBatchSize);
void freeNetwork(Network *network);
void printNetwork(Network *network);
int persistNetwork(Network *network, const char *path);
void fit(Network *network, Dataset *dataset);
void classify(Network *network, Dataset *dataset);
void freeDataset(Dataset *dataset);

#endif
