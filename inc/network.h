#ifndef NETWORK_H
#define NETWORK_H

#include <stdlib.h>

typedef struct
{
    float **neurons;
    float **weights;
    float **biases;
    size_t *layersSizes;
    size_t layerCount;
    size_t maxLayerSize;
    size_t totalNeurons;
    size_t totalSynapses;
    float learningRate;
} NN;

typedef struct
{
    float **inputs;
    float **groundTruths;
    size_t size;
    size_t miniBatchSize;
} Batch;

NN *constructNetwork(int layerCount, size_t *layersSizes, float learningRate);
void freeNetwork(NN *network);
void printNN(NN *network);
int persistNetwork(NN *network, const char *path);

#endif
