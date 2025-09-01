#include "network.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

NN *constructNetwork(int layerCount, size_t *layersSizes, float learningRate)
{
    srand(time(NULL));

    NN *network = calloc(1, sizeof(NN));
    network->layerCount = layerCount;
    network->layersSizes = malloc(layerCount * sizeof(size_t));
    network->neurons = malloc(layerCount * sizeof(float *));
    network->weights = malloc((layerCount - 1) * sizeof(float *));
    network->biases = malloc((layerCount - 1) * sizeof(float *));
    network->learningRate = learningRate;

    for (int l = 0; l < layerCount; l++)
    {
        network->layersSizes[l] = layersSizes[l];
        network->neurons[l] = calloc(layersSizes[l], sizeof(float));
        network->totalNeurons += layersSizes[l];

        if (network->maxLayerSize < layersSizes[l])
            network->maxLayerSize = layersSizes[l];

        if (l > 0)
        {
            network->biases[l - 1] = calloc(layersSizes[l], sizeof(float));
            size_t weights = layersSizes[l] * layersSizes[l - 1];
            network->totalSynapses += weights;
            network->weights[l - 1] = malloc(weights * sizeof(float));

            for (int n = 0; n < layersSizes[l]; n++)
            {
                for (int prevN = 0; prevN < network->layersSizes[l - 1];
                     prevN++)
                {
                    float d = (float)(rand());
                    while (d == 0)
                        d = (float)(rand());
                    float w = (float)(rand()) / d;
                    w = fmodf(w, 3.0f) - 1.0f;
                    network->weights[l - 1][n * layersSizes[l - 1] + prevN] = w;
                }
            }
        }
    }

    return network;
}

void freeNetwork(NN *network)
{
    free(network->neurons[0]);
    for (int i = 1; i < network->layerCount; i++)
    {
        free(network->neurons[i]);
        free(network->weights[i - 1]);
        free(network->biases[i - 1]);
    }
    free(network->neurons);
    free(network->weights);
    free(network->biases);
    free(network->layersSizes);
    free(network);
}

void printNN(NN *nn)
{
    printf("Neural Network Structure:\n");
    printf("Layer count: %zu\n", nn->layerCount);
    printf("Max layer size: %zu\n", nn->maxLayerSize);
    printf("Total neurons: %zu\n", nn->totalNeurons);
    printf("Total synapses: %zu\n", nn->totalSynapses);
    printf("\n");

    printf("Layer sizes:\n");
    for (size_t l = 0; l < nn->layerCount; l++)
    {
        printf("  Layer %zu: %zu neurons\n", l, nn->layersSizes[l]);
    }
    printf("\n");

    printf("Neurons activations:\n");
    for (size_t l = 0; l < nn->layerCount; l++)
    {
        printf("  Layer %zu: ", l);
        for (size_t n = 0; n < nn->layersSizes[l]; n++)
        {
            printf("%f ", nn->neurons[l][n]);
        }
        printf("\n");
    }
    printf("\n");

    // Print biases
    printf("Biases:\n");
    for (size_t l = 1; l < nn->layerCount; l++)
    {
        printf("  Layer %zu: ", l);
        for (size_t n = 0; n < nn->layersSizes[l]; n++)
        {
            printf("%f ", nn->biases[l - 1][n]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Weights:\n");
    for (size_t l = 1; l < nn->layerCount; l++)
    {
        printf("  Weights from Layer %zu to Layer %zu:\n", l - 1, l);
        for (size_t j = 0; j < nn->layersSizes[l]; j++)
        {
            printf("    Neuron %zu: ", j);
            for (size_t k = 0; k < nn->layersSizes[l - 1]; k++)
            {
                printf("%f ",
                       nn->weights[l - 1][j * nn->layersSizes[l - 1] + k]);
            }
            printf("\n");
        }
    }
}

int persistNetwowrk(NN *network, const char *path)
{
    FILE *file = fopen(path, "wb");

    if (!file)
        return 0;

    fwrite(&network->learningRate, sizeof(float), 1, file);
    fwrite(&network->layerCount, sizeof(size_t), 1, file);
    for (int i = 0; i < network->layerCount; i++)
    {
        fwrite(&network->layersSizes[i], sizeof(size_t), 1, file);
        fwrite(network->neurons[i], sizeof(float), network->layersSizes[i],
               file);
    }
    for (int i = 1; i < network->layerCount; i++)
        fwrite(network->biases[i], sizeof(float), network->layersSizes[i],
               file);
    for (int i = 1; i < network->layerCount; i++)
        fwrite(network->weights[i], sizeof(float),
               network->layersSizes[i] * network->layersSizes[i - 1], file);

    fclose(file);
    return 1;
}
