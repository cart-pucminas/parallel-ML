#include "sgd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "network.h"
#include "util.h"

#define IDX2(y, x, nX) ((y) * (nX) + (x))

#define TANH_F(x) (x)
#define RELU_F(x) (x)
#define SOFTMAX_F(x) (x)

#define D_TANH_F(x) (x)
#define D_RELU_F(x) (x)
#define D_SOFTMAX_F(x) (x)

#define PARALLEL_STRATEGY "dynamic parallel for in feed forward outer loop"

void feedForward(NN *network, float **dActZ, ActivationFunction activation)
{
    switch (activation)
    {
    case SIGMOID:
        for (int i = 1; i < network->layerCount; i++)
        {
            int rows = network->layersSizes[i],
                cols = network->layersSizes[i - 1];
            float *l = network->neurons[i], *prevL = network->neurons[i - 1],
                  *w = network->weights[i - 1], *b = network->biases[i - 1];

            #pragma omp parallel for schedule(dynamic)
            for (int j = 0; j < rows; j++)
            {
                l[j] = 0;
                for (int k = 0; k < cols; k++)
                    l[j] += prevL[k] * w[IDX2(j, k, cols)];
                float a = 1.0f / 1.0f + expf(-(l[j] + b[j]));
                if (dActZ != NULL)
                    dActZ[i - 1][j] = a * (1.0f - a);
                l[j] = a;
            }
        }
        break;
    case TANH:
        for (int i = 1; i < network->layerCount; i++)
        {
            int rows = network->layersSizes[i],
                cols = network->layersSizes[i - 1];
            float *l = network->neurons[i], *prevL = network->neurons[i - 1],
                  *w = network->weights[i - 1], *b = network->biases[i - 1];

            #pragma omp parallel for schedule(dynamic)
            for (int j = 0; j < rows; j++)
            {
                l[j] = 0;
                for (int k = 0; k < cols; k++)
                    l[j] += prevL[k] * w[IDX2(j, k, cols)];
                if (dActZ != NULL)
                    dActZ[i - 1][j] = D_TANH_F(l[j] + b[j]);
                l[j] = TANH_F(l[j] + b[j]);
            }
        }
        break;
    case RELU:
        for (int i = 1; i < network->layerCount; i++)
        {
            int rows = network->layersSizes[i],
                cols = network->layersSizes[i - 1];
            float *l = network->neurons[i], *prevL = network->neurons[i - 1],
                  *w = network->weights[i - 1], *b = network->biases[i - 1];

            #pragma omp parallel for schedule(dynamic)
            for (int j = 0; j < rows; j++)
            {
                l[j] = 0;
                for (int k = 0; k < cols; k++)
                    l[j] += prevL[k] * w[IDX2(j, k, cols)];
                if (dActZ != NULL)
                    dActZ[i - 1][j] = D_RELU_F(l[j] + b[j]);
                l[j] = RELU_F(l[j] + b[j]);
            }
        }
        break;
    case SOFTMAX:
        for (int i = 1; i < network->layerCount; i++)
        {
            int rows = network->layersSizes[i],
                cols = network->layersSizes[i - 1];
            float *l = network->neurons[i], *prevL = network->neurons[i - 1],
                  *w = network->weights[i - 1], *b = network->biases[i - 1];

            #pragma omp parallel for schedule(dynamic)
            for (int j = 0; j < rows; j++)
            {
                l[j] = 0;
                for (int k = 0; k < cols; k++)
                    l[j] += prevL[k] * w[IDX2(j, k, cols)];
                if (dActZ != NULL)
                    dActZ[i - 1][j] = D_SOFTMAX_F(l[j] + b[j]);
                l[j] = SOFTMAX_F(l[j] + b[j]);
            }
        }
        break;
    }
}

void backPropagation(NN *network, float *groundTruth, float *nablaW,
                     float *nablaB, float **dZActivation, float **partials)
{
    float *w_p = &nablaW[network->totalSynapses - 1],
          *b_p = &nablaB[network->totalNeurons - 1];

    for (int i = network->layersSizes[network->layerCount - 1] - 1; i > -1; i--)
    {
        *b_p = partials[network->layerCount - 2][i] =
            (dZActivation[network->layerCount - 2][i]) * 2 *
            (network->neurons[network->layerCount - 1][i] - groundTruth[i]);
        b_p--;

        for (int j = network->layersSizes[network->layerCount - 2] - 1; j > -1;
             j--)
        {
            *w_p = network->neurons[network->layerCount - 2][j] *
                   partials[network->layerCount - 2][i];
            w_p--;
        }
    }

    for (int i = network->layerCount - 2; i > 0; i--)
    {
        for (int j = network->layersSizes[i] - 1; j > -1; j--)
        {
            float partial = 0;

            for (int k = network->layersSizes[i + 1] - 1; k > -1; k--)
            {
                partial +=
                    network->weights[i][IDX2(k, j, network->layersSizes[i])] *
                    partials[i][k];
            }

            *b_p = partials[i - 1][j] = dZActivation[i - 1][j] * partial;
            b_p--;

            for (int k = network->layersSizes[i - 1] - 1; k > -1; k--)
            {
                *w_p = network->neurons[i - 1][k] * partials[i - 1][j];
                w_p--;
            }
        }
    }
}

void shuffle(Batch *batch)
{
    srand(time(NULL));

    for (int i = 0; i < batch->size; i++)
    {
        int j = rand() % batch->size;
        float *input = batch->inputs[i], *groundTruth = batch->groundTruths[i];
        batch->inputs[i] = batch->inputs[j];
        batch->groundTruths[i] = batch->groundTruths[j];
        batch->inputs[j] = input;
        batch->groundTruths[j] = groundTruth;
    }
}

void sgd_learn(NN *network, Batch *batch, ActivationFunction activation)
{
    float *localNablaW = malloc(network->totalSynapses * sizeof(float));
    float *nablaW = malloc(network->totalSynapses * sizeof(float));
    float *localNablaB = malloc(network->totalNeurons * sizeof(float));
    float *nablaB = malloc(network->totalNeurons * sizeof(float));
    float **partials = malloc((network->layerCount - 1) * sizeof(float *));
    float **dActZ = malloc((network->layerCount - 1) * sizeof(float *));
    for (int i = 1; i < network->layerCount; i++)
    {
        partials[i - 1] = malloc(network->layersSizes[i] * sizeof(float));
        dActZ[i - 1] = malloc(network->layersSizes[i] * sizeof(float));
    }

    size_t miniBatch_i = 0;
    while (miniBatch_i < batch->size)
    {
        for (int i = 0; i < network->totalSynapses; i++)
            nablaW[i] = 0;
        for (int i = 0; i < network->totalNeurons; i++)
            nablaB[i] = 0;

        int actualBatchSize = 0;
        for (actualBatchSize = 0; actualBatchSize < batch->miniBatchSize &&
                                  miniBatch_i < batch->size;
             actualBatchSize++, miniBatch_i++)
        {
            for (int j = 0; j < network->layersSizes[0]; j++)
                network->neurons[0][j] = batch->inputs[miniBatch_i][j];

            feedForward(network, dActZ, activation);
            backPropagation(network, batch->groundTruths[miniBatch_i],
                            localNablaW, localNablaB, dActZ, partials);

            for (int k = 0; k < network->totalSynapses; k++)
                nablaW[k] += localNablaW[k];
            for (int k = 0; k < network->totalNeurons; k++)
                nablaB[k] += localNablaB[k];
        }

        float eta = network->learningRate / (float)actualBatchSize,
              *nW_p = nablaW, *nB_p = nablaB;

        for (int i = 1; i < network->layerCount; i++)
        {
            int totalW = network->layersSizes[i] * network->layersSizes[i - 1];

            for (int j = 0; j < totalW; j++)
            {
                network->weights[i - 1][j] -= eta * (*nW_p);
                nW_p++;
            }

            for (int j = 0; j < network->layersSizes[i]; j++)
            {
                network->biases[i - 1][j] -= eta * (*nB_p);
                nB_p++;
            }
        }

        printf("%ld out of %ld batches\n", miniBatch_i / batch->miniBatchSize,
               batch->size / batch->miniBatchSize);
        if ((miniBatch_i + 1) < batch->size)
            CLRLINE;
    }

    free(nablaW);
    free(nablaB);
    free(localNablaW);
    free(localNablaB);
    for (int i = 1; i < network->layerCount; i++)
    {
        free(partials[i - 1]);
        free(dActZ[i - 1]);
    }
    free(partials);
    free(dActZ);
}

void sgd_classify(NN *network, Batch *batch, ActivationFunction activation)
{
    size_t hits = 0;
    for (size_t i = 0; i < batch->size; i++)
    {
        for (int j = 0; j < network->layersSizes[0]; j++)
            network->neurons[0][j] = batch->inputs[i][j];

        feedForward(network, NULL, activation);
        float max = -INFINITY;
        int maxIndex = 0, trueMaxIndex = 0;
        for (size_t j = 0; j < network->layersSizes[network->layerCount - 1];
             j++)
        {
            if (network->neurons[network->layerCount - 1][j] > max)
            {
                max = network->neurons[network->layerCount - 1][j];
                maxIndex = j;
            }

            if (batch->groundTruths[i][j] == 1)
                trueMaxIndex = j;
        }
        if (maxIndex == trueMaxIndex)
            hits++;
    }

    printf("%ld/%ld (%.2f%%)\n", hits, batch->size,
           ((float)hits / batch->size) * 100.0f);
}
