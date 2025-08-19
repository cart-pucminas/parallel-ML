#include "perceptron.h"

#include <stdio.h>

#include "network.h"
#include "sgd.h"

Batch *toNNBatch(Dataset *dataset, size_t miniBatchSize)
{
    Batch *batch = malloc(sizeof(Batch));
    batch->size = dataset->size;
    batch->miniBatchSize = miniBatchSize;
    batch->inputs = malloc(dataset->size * sizeof(float *));
    batch->groundTruths = malloc(dataset->size * sizeof(float *));
    for (int i = 0; i < dataset->size; i++)
    {
        batch->inputs[i] =
            malloc(dataset->imageWidth * dataset->imageHeight * sizeof(float));
        for (int j = 0; j < dataset->imageWidth * dataset->imageHeight; j++)
            batch->inputs[i][j] = dataset->images[i][j] / 255.0f;

        batch->groundTruths[i] = malloc(10 * sizeof(float));
        for (int j = 0; j < 10; j++)
            batch->groundTruths[i][j] = (float)(j == dataset->labels[i]);
    }
    return batch;
}

void freeBatch(Batch *batch)
{
    for (int i = 0; i < batch->size; i++)
    {
        free(batch->inputs[i]);
        free(batch->groundTruths[i]);
    }
    free(batch->inputs);
    free(batch->groundTruths);
    free(batch);
}

void init(Params *params, Dataset *learningDataset,
          Dataset *classificationDataset)
{
    size_t *sizes = malloc((params->hiddenLayerCount + 2) * sizeof(size_t));

    sizes[0] = 784;
    sizes[params->hiddenLayerCount + 1] = 10;
    for (int i = 0; i < params->hiddenLayerCount; i++)
        sizes[i + 1] = params->hiddenLayerSizes[i];

    NN *network = constructNetwork(params->hiddenLayerCount + 2, sizes,
                                   params->learningRate);

    Batch *learningBatch = toNNBatch(learningDataset, params->miniBatchSize);
    Batch *classificationBatch =
        toNNBatch(classificationDataset, params->miniBatchSize);

    for (int i = 0; i < params->epochs; i++)
    {
        printf("Epoch %d\n", i + 1);
        sgd_learn(network, learningBatch, params->activation);
        sgd_classify(network, classificationBatch, params->activation);
    }
}
