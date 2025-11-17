#include <omp.h>
#include <stdio.h>
#include <string.h>

#include "dataloader.h"
#include "dataset.h"
#include "mlp.h"

const char grayscaleMap[] = ".:-=+*#%@";

int main(int argc, char **argv)
{
    omp_set_num_threads(1);
    char *labels =
             "/home/mateus/repos/perceptron/input/train-labels.idx1-ubyte",
         *images =
             "/home/mateus/repos/perceptron/input/train-images.idx3-ubyte";

    Dataset *learningDataset = loadDataset(labels, images);
    if (learningDataset == NULL)
    {
        printf("Dataset error: %s", dataset_getError());
        return 1;
    }

    labels = "/home/mateus/repos/perceptron/input/t10k-labels.idx1-ubyte",
    images = "/home/mateus/repos/perceptron/input/t10k-images.idx3-ubyte";

    Dataset *classificationDataset = loadDataset(labels, images);
    if (classificationDataset == NULL)
    {
        printf("Dataset error: %s", dataset_getError());
        return 1;
    }

    unsigned int layers[] = {28 * 28, 256, 128, 10};
    Network *n = constructNetwork(SIGMOID, 10, 4, layers, 0.5, 128);
    fit(n, learningDataset);
    classify(n, classificationDataset);

    freeDataset(learningDataset);
    freeDataset(classificationDataset);
    freeNetwork(n);

    printf("all done :D\n");
    return 0;
}
