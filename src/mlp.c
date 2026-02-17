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

Network *constructNetwork(size_t epochs, size_t layerCount, size_t *layersSizes,
                          float learningRate, size_t miniBatchSize)
{
    srand(time(NULL));

    Network *network = calloc(1, sizeof(Network));
    network->epochs = epochs;
    network->layerCount = layerCount;
    network->layersSizes = malloc(layerCount * sizeof(size_t));
    network->weights = malloc((layerCount - 1) * sizeof(float *));
    network->biases = malloc((layerCount - 1) * sizeof(float *));
    network->learningRate = learningRate;
    network->miniBatchSize = miniBatchSize;

    for (size_t l = 0; l < layerCount; l++)
    {
        network->layersSizes[l] = layersSizes[l];
        network->totalNeurons += layersSizes[l];

        if (network->maxLayerSize < layersSizes[l])
            network->maxLayerSize = layersSizes[l];

        if (l > 0)
        {
            network->biases[l - 1] = calloc(layersSizes[l], sizeof(float));
            size_t weights = layersSizes[l] * layersSizes[l - 1];
            network->totalSynapses += weights;
            network->weights[l - 1] = malloc(weights * sizeof(float));

            for (size_t n = 0; n < layersSizes[l]; n++)
            {
                for (size_t prevN = 0; prevN < network->layersSizes[l - 1];
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

void freeNetwork(Network *network)
{
    for (size_t i = 1; i < network->layerCount; i++)
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
    printf("Layer count: %d\n", nn->layerCount);
    printf("Max layer size: %d\n", nn->maxLayerSize);
    printf("Total neurons: %zu\n", nn->totalNeurons);
    printf("Total synapses: %zu\n", nn->totalSynapses);
    printf("\n");

    printf("Layer sizes:\n");
    for (size_t l = 0; l < nn->layerCount; l++)
    {
        printf("  Layer %zu: %d neurons\n", l, nn->layersSizes[l]);
    }
    printf("\n");

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

int persistNetwowrk(Network *network, const char *path)
{
    FILE *file = fopen(path, "wb");

    if (!file)
        return 0;

    fwrite(&network->learningRate, sizeof(float), 1, file);
    fwrite(&network->layerCount, sizeof(size_t), 1, file);
    for (size_t i = 0; i < network->layerCount; i++)
    {
        fwrite(&network->layersSizes[i], sizeof(size_t), 1, file);
    }
    for (size_t i = 1; i < network->layerCount; i++)
        fwrite(network->biases[i], sizeof(float), network->layersSizes[i],
               file);
    for (size_t i = 1; i < network->layerCount; i++)
        fwrite(network->weights[i], sizeof(float),
               network->layersSizes[i] * network->layersSizes[i - 1], file);

    fclose(file);
    return 1;
}

void feedForward(Network *network, float **neurons, float **dActZ)
{
    for (size_t i = 1; i < network->layerCount; i++)
    {
        int rows = network->layersSizes[i], cols = network->layersSizes[i - 1];
        float *l = neurons[i], *prevL = neurons[i - 1],
              *w = network->weights[i - 1], *b = network->biases[i - 1];

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

            if (dActZ != NULL)
                dActZ[i - 1][j] = a * (1.0f - a);

            l[j] = a;
        }
    }
}

void backPropagation(Network *network, float *groundTruth,
                     ThreadWorkspace *workspace, float *nablaW, float *nablaB)
{
#ifdef INTRA_LAYER
    size_t ultimateLayerSize = network->layersSizes[network->layerCount - 1];
    size_t penultimateLayerSize = network->layersSizes[network->layerCount - 2];

    size_t offsetB = (network->totalNeurons - ultimateLayerSize);
    size_t offsetW =
        (network->totalSynapses - (ultimateLayerSize * penultimateLayerSize));

#pragma omp parallel for
    for (size_t i = ultimateLayerSize - 1; i >= 0; i--)
    {
        nablaB[offsetB + i] += workspace->partials[network->layerCount - 2][i] =
            (workspace->dActZ[network->layerCount - 2][i]) * 2 *
            (workspace->neurons[network->layerCount - 1][i] - groundTruth[i]);

        for (size_t j = penultimateLayerSize - 1; j > -1; j--)
        {
            nablaW[offsetW + (i * penultimateLayerSize) + j] +=
                workspace->neurons[network->layerCount - 2][j] *
                workspace->partials[network->layerCount - 2][i];
        }
    }

    for (size_t i = network->layerCount - 2; i > 0; i--)
    {
        offsetB -= network->layersSizes[i];
        offsetW -= network->layersSizes[i] * network->layersSizes[i - 1];

#pragma omp parallel for
        for (size_t j = 0; j < network->layersSizes[i]; j++)
        {
            float partial = 0;

#ifdef VECTORIZED
#pragma omp simd reduction(+ : partial)
#endif
            for (size_t k = 0; k < network->layersSizes[i + 1]; k++)
            {
                partial +=
                    network->weights[i][IDX2(k, j, network->layersSizes[i])] *
                    workspace->partials[i][k];
            }

            nablaB[offsetB + j] += workspace->partials[i - 1][j] =
                workspace->dActZ[i - 1][j] * partial;

            for (size_t k = 0; k < network->layersSizes[i - 1]; k++)
            {
                nablaW[offsetW + (j * network->layersSizes[i - 1]) + k] +=
                    workspace->neurons[i - 1][k] *
                    workspace->partials[i - 1][j];
            }
        }
    }
#else
    float *w_p = &nablaW[network->totalSynapses - 1];
    float *b_p = &nablaB[network->totalNeurons - 1];

    for (int i = network->layersSizes[network->layerCount - 1] - 1; i > -1; i--)
    {
        *b_p += workspace->partials[network->layerCount - 2][i] =
            (workspace->dActZ[network->layerCount - 2][i]) * 2 *
            (workspace->neurons[network->layerCount - 1][i] - groundTruth[i]);
        b_p--;

        for (int j = network->layersSizes[network->layerCount - 2] - 1; j > -1;
             j--)
        {
            *w_p += workspace->neurons[network->layerCount - 2][j] *
                    workspace->partials[network->layerCount - 2][i];
            w_p--;
        }
    }

    for (int i = network->layerCount - 2; i > 0; i--)
    {
        for (int j = network->layersSizes[i] - 1; j > -1; j--)
        {
            float partial = 0;

#ifdef VECTORIZED
#pragma omp simd reduction(+ : partial)
#endif
            for (int k = network->layersSizes[i + 1] - 1; k > -1; k--)
            {
                partial +=
                    network->weights[i][IDX2(k, j, network->layersSizes[i])] *
                    workspace->partials[i][k];
            }

            *b_p += workspace->partials[i - 1][j] =
                workspace->dActZ[i - 1][j] * partial;
            b_p--;

            for (int k = network->layersSizes[i - 1] - 1; k > -1; k--)
            {
                *w_p += workspace->neurons[i - 1][k] *
                        workspace->partials[i - 1][j];
                w_p--;
            }
        }
    }
#endif
}

void fit(Network *network, Dataset *dataset)
{
    Timer timer;

    profile_start(&timer);

    int maxThreads = omp_get_max_threads();

    float *nablaW = malloc(network->totalSynapses * sizeof(float));
    float *nablaB = malloc(network->totalNeurons * sizeof(float));

    ThreadWorkspace *workspaces = malloc(maxThreads * sizeof(ThreadWorkspace));

    for (int i = 0; i < maxThreads; i++)
    {
        workspaces[i].neurons = malloc(network->layerCount * sizeof(float *));

        for (size_t j = 0; j < network->layerCount; j++)
        {
            workspaces[i].neurons[j] =
                malloc(network->layersSizes[j] * sizeof(float));
        }

        workspaces[i].partials =
            malloc((network->layerCount - 1) * sizeof(float *));
        workspaces[i].dActZ =
            malloc((network->layerCount - 1) * sizeof(float *));

        for (size_t j = 1; j < network->layerCount; j++)
        {
            workspaces[i].partials[j - 1] =
                malloc(network->layersSizes[j] * sizeof(float));
            workspaces[i].dActZ[j - 1] =
                malloc(network->layersSizes[j] * sizeof(float));
        }
    }

    size_t miniCount = dataset->size / network->miniBatchSize + 1;
    for (size_t e = 0; e < network->epochs; e++)
    {
        for (size_t miniBatchStart = 0; miniBatchStart < dataset->size;
             miniBatchStart += network->miniBatchSize)
        {
            memset(nablaW, 0, network->totalSynapses * sizeof(float));
            memset(nablaB, 0, network->totalNeurons * sizeof(float));

            size_t miniBatchEnd = miniBatchStart + network->miniBatchSize;
            if (miniBatchEnd > dataset->size)
                miniBatchEnd = dataset->size;

            int trueMiniSize = miniBatchEnd - miniBatchStart;

            if (trueMiniSize != 0)
            {
#ifdef INTER_SAMPLE
#pragma omp parallel for reduction(+ : nablaW[ : network->totalSynapses],      \
                                       nablaB[ : network->totalNeurons])
#endif
                for (int i = 0; i < trueMiniSize; i++)
                {
                    int t = omp_get_thread_num();

                    for (size_t j = 0; j < network->layersSizes[0]; j++)
                        workspaces[t].neurons[0][j] =
                            dataset->inputs[i + miniBatchStart][j];

                    feedForward(network, workspaces[t].neurons,
                                workspaces[t].dActZ);
                    backPropagation(network,
                                    dataset->groundTruths[i + miniBatchStart],
                                    &workspaces[t], nablaW, nablaB);
                }

                float eta = network->learningRate / (float)trueMiniSize,
                      *nW_p = nablaW, *nB_p = nablaB;

                for (size_t i = 1; i < network->layerCount; i++)
                {
                    int totalW =
                        network->layersSizes[i] * network->layersSizes[i - 1];

#ifdef INTER_SAMPLE
#pragma omp parallel for
#endif
                    for (int j = 0; j < totalW; j++)
                    {
                        network->weights[i - 1][j] -= eta * (*nW_p);
                        nW_p++;
                    }

#ifdef INTER_SAMPLE
#pragma omp parallel for
#endif
                    for (size_t j = 0; j < network->layersSizes[i]; j++)
                    {
                        network->biases[i - 1][j] -= eta * (*nB_p);
                        nB_p++;
                    }
                }

                printf("Epoch %ld - %ld out of %ld batches\n", e + 1,
                       miniBatchStart / network->miniBatchSize + 1, miniCount);
                if (miniBatchEnd < dataset->size)
                    CLRLINE;
            }
        }
    }

    free(nablaW);
    free(nablaB);
    for (int i = 0; i < maxThreads; i++)
    {
        for (size_t j = 0; j < network->layerCount; j++)
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
    for (size_t i = 0; i < network->layerCount; i++)
        neurons[i] = malloc(network->layersSizes[i] * sizeof(float));

    size_t hits = 0;
    for (size_t i = 0; i < dataset->size; i++)
    {
        for (size_t j = 0; j < network->layersSizes[0]; j++)
            neurons[0][j] = dataset->inputs[i][j];

        feedForward(network, neurons, NULL);

        float max = -INFINITY;
        int maxIndex = 0, trueMaxIndex = 0;
        for (size_t j = 0; j < network->layersSizes[network->layerCount - 1];
             j++)
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

    printf("%ld/%ld (%.2f%%)\n", hits, dataset->size,
           ((float)hits / dataset->size) * 100.0f);

    for (size_t i = 0; i < network->layerCount; i++)
        free(neurons[i]);
    free(neurons);
}

void freeDataset(Dataset *dataset)
{
    for (size_t i = 0; i < dataset->size; i++)
    {
        free(dataset->inputs[i]);
        free(dataset->groundTruths[i]);
    }
    free(dataset->inputs);
    free(dataset->groundTruths);
    free(dataset);
}
