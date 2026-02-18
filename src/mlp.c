#include "mlp.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "profiler.h"
#include "util.h"

#define IDX2(y, x, nX) ((y) * (nX) + (x))

#define TANH_F(x) (x)
#define RELU_F(x) (x)

#define D_TANH_F(x) (x)
#define D_RELU_F(x) (x)

typedef struct
{
    float **neurons;
    float **partials;
    float **dActZ;
} ThreadWorkspace;

Network *constructNetwork(unsigned int epochs, unsigned int layerCount,
                          unsigned int *layersSizes, float learningRate,
                          unsigned int miniBatchSize, unsigned int seed)
{
    srand(seed);

    Network *network = calloc(1, sizeof(Network));
    network->epochs = epochs;
    network->layerCount = layerCount;
    network->layersSizes = malloc(layerCount * sizeof(unsigned int));
    network->weights = malloc((layerCount - 1) * sizeof(float *));
    network->biases = malloc((layerCount - 1) * sizeof(float *));
    network->learningRate = learningRate;
    network->miniBatchSize = miniBatchSize;

    network->layersSizes[0] = layersSizes[0];
    network->totalNeurons += layersSizes[0];
    network->maxLayerSize = layersSizes[0];

    for (unsigned int l = 1; l < layerCount; l++)
    {
        network->layersSizes[l] = layersSizes[l];
        network->totalNeurons += layersSizes[l];

        if (network->maxLayerSize < layersSizes[l])
            network->maxLayerSize = layersSizes[l];

        network->biases[l - 1] = calloc(layersSizes[l], sizeof(float));
        unsigned int weights = layersSizes[l] * layersSizes[l - 1];
        network->totalSynapses += weights;
        network->weights[l - 1] = malloc(weights * sizeof(float));

        for (unsigned int n = 0; n < layersSizes[l]; n++)
        {
            for (unsigned int prevN = 0; prevN < network->layersSizes[l - 1];
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

    return network;
}

void freeNetwork(Network *network)
{
    for (unsigned int i = 1; i < network->layerCount; i++)
    {
        free(network->weights[i - 1]);
        free(network->biases[i - 1]);
    }
    free(network->weights);
    free(network->biases);
    free(network->layersSizes);
    free(network);
}

void printNetwork(Network *nn)
{
    printf("Neural Network Structure:\n");
    printf("Layer count: %u\n", nn->layerCount);
    printf("Max layer size: %u\n", nn->maxLayerSize);
    printf("Total neurons: %zu\n", nn->totalNeurons);
    printf("Total synapses: %zu\n", nn->totalSynapses);
    printf("\n");

    printf("Layer sizes:\n");
    for (unsigned int l = 0; l < nn->layerCount; l++)
    {
        printf("  Layer %u: %u neurons\n", l, nn->layersSizes[l]);
    }
    printf("\n");

    printf("Biases:\n");
    for (unsigned int l = 1; l < nn->layerCount; l++)
    {
        printf("  Layer %u: ", l);
        for (unsigned int n = 0; n < nn->layersSizes[l]; n++)
        {
            printf("%f ", nn->biases[l - 1][n]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Weights:\n");
    for (unsigned int l = 1; l < nn->layerCount; l++)
    {
        printf("  Weights from Layer %u to Layer %u:\n", l - 1, l);
        for (unsigned int j = 0; j < nn->layersSizes[l]; j++)
        {
            printf("    Neuron %u: ", j);
            for (unsigned int k = 0; k < nn->layersSizes[l - 1]; k++)
            {
                printf("%f ",
                       nn->weights[l - 1][j * nn->layersSizes[l - 1] + k]);
            }
            printf("\n");
        }
    }
}

int persistNetwowrk(Network *network, const char *path)
{
    FILE *file = fopen(path, "wb");

    if (!file)
        return 0;

    fwrite(&network->learningRate, sizeof(float), 1, file);
    fwrite(&network->layerCount, sizeof(unsigned int), 1, file);
    for (unsigned int i = 0; i < network->layerCount; i++)
    {
        fwrite(&network->layersSizes[i], sizeof(unsigned int), 1, file);
    }
    for (unsigned int i = 1; i < network->layerCount; i++)
        fwrite(network->biases[i], sizeof(float), network->layersSizes[i],
               file);
    for (unsigned int i = 1; i < network->layerCount; i++)
        fwrite(network->weights[i], sizeof(float),
               network->layersSizes[i] * network->layersSizes[i - 1], file);

    fclose(file);
    return 1;
}

void feedForward(Network *network, float **neurons, float **dActZ)
{
    for (unsigned int i = 1; i < network->layerCount; i++)
    {
        int rows = network->layersSizes[i], cols = network->layersSizes[i - 1];

        float *restrict l = neurons[i];
        float *restrict prevL = neurons[i - 1];
        float *restrict w = network->weights[i - 1];
        float *restrict b = network->biases[i - 1];

#ifdef INTRA_LAYER
#pragma omp parallel for
#endif
        for (int j = 0; j < rows; j++)
        {
            float z = 0;

#ifdef VECTORIZED
#pragma omp simd reduction(+ : z)
#endif
            for (int k = 0; k < cols; k++)
                z += prevL[k] * w[IDX2(j, k, cols)];

            z += b[j];
            float a = 1.0f / (1.0f + expf(-z));

            dActZ[i - 1][j] = a * (1.0f - a);

            l[j] = a;
        }
    }
}

void backPropagation(Network *network, float *groundTruth,
                     ThreadWorkspace *workspace, float *restrict nablaW,
                     float *restrict nablaB)
{
    unsigned int ultimateLayerSize =
        network->layersSizes[network->layerCount - 1];
    unsigned int penultimateLayerSize =
        network->layersSizes[network->layerCount - 2];

    unsigned int offsetB = (network->totalNeurons - ultimateLayerSize);
    unsigned int offsetW =
        (network->totalSynapses - (ultimateLayerSize * penultimateLayerSize));

#ifdef INTRA_LAYER
#pragma omp parallel for
#endif
    for (long int i = ultimateLayerSize - 1; i >= 0; i--)
    {
        nablaB[offsetB + i] += workspace->partials[network->layerCount - 2][i] =
            (workspace->dActZ[network->layerCount - 2][i]) * 2 *
            (workspace->neurons[network->layerCount - 1][i] - groundTruth[i]);

        for (long int j = penultimateLayerSize - 1; j > -1; j--)
        {
            nablaW[offsetW + (i * penultimateLayerSize) + j] +=
                workspace->neurons[network->layerCount - 2][j] *
                workspace->partials[network->layerCount - 2][i];
        }
    }

    for (long int i = network->layerCount - 2; i > 0; i--)
    {
        offsetB -= network->layersSizes[i];
        offsetW -= network->layersSizes[i] * network->layersSizes[i - 1];

#ifdef INTRA_LAYER
#pragma omp parallel for
#endif
        for (unsigned int j = 0; j < network->layersSizes[i]; j++)
        {
            float partial = 0;

#ifdef VECTORIZED
#pragma omp simd reduction(+ : partial)
#endif
            for (unsigned int k = 0; k < network->layersSizes[i + 1]; k++)
            {
                partial +=
                    network->weights[i][IDX2(k, j, network->layersSizes[i])] *
                    workspace->partials[i][k];
            }

            nablaB[offsetB + j] += workspace->partials[i - 1][j] =
                workspace->dActZ[i - 1][j] * partial;

            for (unsigned int k = 0; k < network->layersSizes[i - 1]; k++)
            {
                nablaW[offsetW + (j * network->layersSizes[i - 1]) + k] +=
                    workspace->neurons[i - 1][k] *
                    workspace->partials[i - 1][j];
            }
        }
    }
}

void fit(Network *network, Dataset *dataset)
{
    Timer timer;

    profile_start(&timer);

    int maxThreads = 1;

#ifndef NO_OMP
    maxThreads = omp_get_max_threads();
#endif

    float *nablaW = malloc(network->totalSynapses * sizeof(float));
    float *nablaB = malloc(network->totalNeurons * sizeof(float));

    ThreadWorkspace *workspaces = malloc(maxThreads * sizeof(ThreadWorkspace));

    for (int i = 0; i < maxThreads; i++)
    {
        workspaces[i].neurons = malloc(network->layerCount * sizeof(float *));

        for (unsigned int j = 0; j < network->layerCount; j++)
        {
            workspaces[i].neurons[j] =
                malloc(network->layersSizes[j] * sizeof(float));
        }

        workspaces[i].partials =
            malloc((network->layerCount - 1) * sizeof(float *));
        workspaces[i].dActZ =
            malloc((network->layerCount - 1) * sizeof(float *));

        for (unsigned int j = 1; j < network->layerCount; j++)
        {
            workspaces[i].partials[j - 1] =
                malloc(network->layersSizes[j] * sizeof(float));
            workspaces[i].dActZ[j - 1] =
                malloc(network->layersSizes[j] * sizeof(float));
        }
    }

    unsigned int miniCount = dataset->size / network->miniBatchSize + 1;
    for (unsigned int e = 0; e < network->epochs; e++)
    {
        for (unsigned int miniBatchStart = 0; miniBatchStart < dataset->size;
             miniBatchStart += network->miniBatchSize)
        {
            memset(nablaW, 0, network->totalSynapses * sizeof(float));
            memset(nablaB, 0, network->totalNeurons * sizeof(float));

            unsigned int miniBatchEnd = miniBatchStart + network->miniBatchSize;

            if (miniBatchEnd > dataset->size)
                miniBatchEnd = dataset->size;

            unsigned int trueMiniSize = miniBatchEnd - miniBatchStart;

#ifdef INTER_SAMPLE
#pragma omp parallel for reduction(+ : nablaW[ : network->totalSynapses],      \
                                       nablaB[ : network->totalNeurons])
#endif
            for (unsigned int i = 0; i < trueMiniSize; i++)
            {
                int t = 0;

#ifndef NO_OMP
                t = omp_get_thread_num();
#endif

                for (unsigned int j = 0; j < network->layersSizes[0]; j++)
                    workspaces[t].neurons[0][j] =
                        dataset->inputs[i + miniBatchStart][j];

                feedForward(network, workspaces[t].neurons,
                            workspaces[t].dActZ);
                backPropagation(network,
                                dataset->groundTruths[i + miniBatchStart],
                                &workspaces[t], nablaW, nablaB);
            }

            float eta = network->learningRate / (float)trueMiniSize;

            unsigned int offsetW = 0, offsetB = 0;
            for (unsigned int i = 1; i < network->layerCount; i++)
            {
                size_t totalW =
                    network->layersSizes[i] * network->layersSizes[i - 1];

#ifdef INTER_SAMPLE
#pragma omp parallel for
#endif
                for (size_t j = 0; j < totalW; j++)
                {
                    network->weights[i - 1][j] -= eta * nablaW[offsetW + j];
                }

#ifdef INTER_SAMPLE
#pragma omp parallel for
#endif
                for (unsigned int j = 0; j < network->layersSizes[i]; j++)
                {
                    network->biases[i - 1][j] -= eta * nablaB[offsetB + j];
                }

                offsetW += totalW;
                offsetB += network->layersSizes[i];
            }

            printf("Epoch %u - %u/%u batches\n", e + 1,
                   miniBatchStart / network->miniBatchSize + 1, miniCount);

            if (miniBatchEnd < dataset->size)
                CLRLINE;
        }
    }

    free(nablaW);
    free(nablaB);
    for (int i = 0; i < maxThreads; i++)
    {
        for (unsigned int j = 0; j < network->layerCount; j++)
        {
            free(workspaces[i].neurons[j]);
            if (j > 0)
            {
                free(workspaces[i].partials[j - 1]);
                free(workspaces[i].dActZ[j - 1]);
            }
        }
        free(workspaces[i].neurons);
        free(workspaces[i].partials);
        free(workspaces[i].dActZ);
    }
    free(workspaces);

    printf("%lf seconds elapsed\n", profile_getElapsed(&timer));
}

void classify(Network *network, Dataset *dataset)
{
    float **neurons = malloc(network->layerCount * sizeof(float *));
    for (unsigned int i = 0; i < network->layerCount; i++)
        neurons[i] = malloc(network->layersSizes[i] * sizeof(float));

    // Dummy activation derivative array because feedForward needs one
    float **dummyDActZ = malloc((network->layerCount - 1) * sizeof(float *));

    for (unsigned int j = 1; j < network->layerCount; j++)
    {
        dummyDActZ[j - 1] = malloc(network->layersSizes[j] * sizeof(float));
    }

    unsigned int hits = 0;
    for (unsigned int i = 0; i < dataset->size; i++)
    {
        for (unsigned int j = 0; j < network->layersSizes[0]; j++)
            neurons[0][j] = dataset->inputs[i][j];

        feedForward(network, neurons, dummyDActZ);

        float max = -INFINITY;
        int maxIndex = 0, trueMaxIndex = 0;
        for (unsigned int j = 0;
             j < network->layersSizes[network->layerCount - 1]; j++)
        {
            if (neurons[network->layerCount - 1][j] > max)
            {
                max = neurons[network->layerCount - 1][j];
                maxIndex = j;
            }

            if (dataset->groundTruths[i][j] == 1)
                trueMaxIndex = j;
        }
        if (maxIndex == trueMaxIndex)
            hits++;
    }

    printf("%d/%zu (%.2f%%)\n", hits, dataset->size,
           ((float)hits / dataset->size) * 100.0f);

    for (unsigned int i = 0; i < network->layerCount; i++)
        free(neurons[i]);
    free(neurons);
}

void freeDataset(Dataset *dataset)
{
    for (unsigned int i = 0; i < dataset->size; i++)
    {
        free(dataset->inputs[i]);
        free(dataset->groundTruths[i]);
    }
    free(dataset->inputs);
    free(dataset->groundTruths);
    free(dataset);
}
