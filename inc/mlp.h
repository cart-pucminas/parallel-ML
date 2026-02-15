#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <stdlib.h>

typedef struct
{
    int epochs;
    float learningRate;
    unsigned int miniBatchSize;
    float **weights;
    float **biases;
    unsigned int *layersSizes;
    unsigned int layerCount;
    unsigned int maxLayerSize;
    size_t totalNeurons;
    size_t totalSynapses;
} Network;

typedef struct
{
    float **inputs;
    float **groundTruths;
    size_t size;
} Dataset;

Network *constructNetwork(int epochs, unsigned int layerCount,
                          unsigned int *layersSizes, float learningRate,
                          unsigned int miniBatchSize);
void freeNetwork(Network *network);
void printNetwork(Network *network);
int persistNetwork(Network *network, const char *path);
void fit(Network *network, Dataset *dataset);
void classify(Network *network, Dataset *dataset);
void freeDataset(Dataset *dataset);

#endif
