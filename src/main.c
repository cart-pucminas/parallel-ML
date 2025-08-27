#include <stdio.h>
#include <string.h>

#include "args.h"
#include "dataloader.h"
#include "dataset.h"
#include "perceptron.h"

const char grayscaleMap[] = ".:-=+*#%@";

int main(int argc, char **argv)
{
    Params *params = parseArgs(argc, argv);

    char *labels = "/home/mateus/repos/perceptron/input/test-label.idx1-ubyte",
         *images = "/home/mateus/repos/perceptron/input/test-image.idx3-ubyte";

    Dataset *learningDataset = loadDataset(labels, images);
    if (learningDataset == NULL)
    {
        printf("Dataset error: %s", dataset_getError());
        return 1;
    }

    labels = "/home/mateus/repos/perceptron/input/test-label.idx1-ubyte",
    images = "/home/mateus/repos/perceptron/input/test-image.idx3-ubyte";

    Dataset *classificationDataset = loadDataset(labels, images);
    if (classificationDataset == NULL)
    {
        printf("Dataset error: %s", dataset_getError());
        return 1;
    }

    init(params, learningDataset, classificationDataset);

    freeDataset(learningDataset);
    freeDataset(classificationDataset);
    freeParams(params);

    printf("all done :D\n");
    return 0;
}
