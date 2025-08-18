#include "perceptron.h"

#include "network.h"
#include "sgd.h"

static ActivationFunction activation = SIGMOID;
static NN *network = NULL;

Batch *toNNBatch(Dataset *dataset)
{
    Batch *batch = malloc(sizeof(Batch));
    batch->size = dataset->size;
    batch->miniBatchSize = 10;
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

void learn(Dataset *dataset)
{
    if (network == NULL)
    {
        size_t sizes[] = {dataset->imageWidth * dataset->imageHeight, 15, 10};
        network = constructNetwork(3, sizes);
    }

    Batch *batch = toNNBatch(dataset);

    sgd_learn(network, batch, activation);

    freeBatch(batch);
}

void classify(Dataset *dataset)
{
    if (network == NULL)
    {
        size_t sizes[] = {dataset->imageWidth * dataset->imageHeight, 10};
        network = constructNetwork(2, sizes);
    }

    Batch *batch = toNNBatch(dataset);

    sgd_classify(network, batch, activation);

    freeBatch(batch);
}
