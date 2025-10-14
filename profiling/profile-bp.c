#include <stdio.h>

#include "network.h"
#include "profiler.h"
#include "sgd.h"

int main(int argc, char **argv)
{
    size_t sizes[] = {784, 128, 128, 10};
    NN *network = constructNetwork(4, sizes, 0.1);

    float *nablaW = malloc(network->totalSynapses * sizeof(float));
    float *nablaB = malloc(network->totalNeurons * sizeof(float));
    float **partials = malloc((network->layerCount - 1) * sizeof(float *));
    float **dActZ = malloc((network->layerCount - 1) * sizeof(float *));
    float *groundTruth = calloc(10, sizeof(float));
    groundTruth[0] = 1;

    for (int i = 1; i < network->layerCount; i++)
    {
        dActZ[i - 1] = malloc(network->layersSizes[i] * sizeof(float));
        partials[i - 1] = malloc(network->layersSizes[i] * sizeof(float));
    }

    feedForward(network, dActZ, SIGMOID);
    backPropagation(network, groundTruth, nablaW, nablaB, dActZ, partials);

    double elapsedSum = 0;
    for (int i = 0; i < 10; i++)
    {
        profile_start();
        double elapsed = profile_getElapsed();
        printf("%.9f\n", elapsed);
        elapsedSum += elapsed;
    }

    printf("%.9f\n", elapsedSum / 10);

    freeNetwork(network);

    return 0;
}
