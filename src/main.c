#include <omp.h>
#include <stdio.h>

#include "dataset.h"
#include "mlp.h"
#include "mnist_dataloader.h"

const char grayscaleMap[] = ".:-=+*#%@";

int main()
{
    omp_set_num_threads(4);
    char *labels = "/input/train-labels.idx1-ubyte",
         *images = "/input/train-images.idx3-ubyte";

    MnistDataset *learningDataset = loadDataset(labels, images);
    if (learningDataset == NULL)
    {
        printf("Dataset error: %s", dataset_getError());
        return 1;
    }

    labels = "/home/mateus/repos/perceptron/input/t10k-labels.idx1-ubyte",
    images = "/home/mateus/repos/perceptron/input/t10k-images.idx3-ubyte";

    MnistDataset *classificationDataset = loadDataset(labels, images);
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
