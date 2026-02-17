#ifndef ROOT_DIR
#define ROOT_DIR "."
#endif

#if defined(NO_OMP) &&                                                         \
    (defined(VECTORIZED) || defined(INTRA_LAYER) || defined(INTER_SAMPLE))
#error "flag NO_OMP should be defined alone"
#endif

#include <omp.h>
#include <stdio.h>

#include "mlp.h"
#include "mnist_dataloader.h"
#include "xor_dataloader.h"

const char grayscaleMap[] = ".:-=+*#%@";

void printFlags()
{
    printf("flags: ");

#ifdef NO_OMP
    printf("NO_OMP ");
#endif

#ifdef VECTORIZED
    printf("VECTORIZED ");
#endif

#ifdef INTRA_LAYER
    printf("INTRA_LAYER ");
#endif

#ifdef INTER_SAMPLE
    printf("INTER_SAMPLE ");
#endif

    printf("\n");
}

int xor()
{
    char *labels = "/input/xor/train-labels",
         *inputs = "/input/xor/train-inputs";

    Dataset *learningDataset = NULL;
    Dataset *classificationDataset = NULL;

    if (!xor_loadDataset(&learningDataset, labels, inputs))
    {
        printf("Dataset error: %s", xor_dataset_getError());
        return 1;
    }

    labels = "/input/xor/test-labels", inputs = "/input/xor/test-inputs";

    if (!xor_loadDataset(&classificationDataset, labels, inputs))
    {
        printf("Dataset error: %s", xor_dataset_getError());
        return 1;
    }

    unsigned int layers[] = {2, 2, 1};
    Network *n = constructNetwork(1, 3, layers, 0.5, 100, 42);

    fit(n, learningDataset);
    classify(n, classificationDataset);

    freeDataset(learningDataset);
    freeDataset(classificationDataset);
    freeNetwork(n);

    printf("all done :D\n");

    return 0;
}

int mnist()
{
    char *labels = "/input/mnist/train-labels.idx1-ubyte",
         *images = "/input/mnist/train-images.idx3-ubyte";

    Dataset *learningDataset = NULL;
    Dataset *classificationDataset = NULL;

    if (!mnist_loadDataset(&learningDataset, labels, images))
    {
        printf("Dataset error: %s", xor_dataset_getError());
        return 1;
    }

    labels = "/input/mnist/t10k-labels.idx1-ubyte",
    images = "/input/mnist/t10k-images.idx3-ubyte";

    if (!mnist_loadDataset(&classificationDataset, labels, images))
    {
        printf("Dataset error: %s", xor_dataset_getError());
        return 1;
    }

    unsigned int layers[] = {28 * 28, 100, 10};
    Network *n = constructNetwork(10, 3, layers, 0.5, 128, 42);

    fit(n, learningDataset);
    classify(n, classificationDataset);

    freeDataset(learningDataset);
    freeDataset(classificationDataset);
    freeNetwork(n);

    printf("all done :D\n");

    return 0;
}

int main()
{
    printFlags();

#ifndef NO_OMP
    omp_set_num_threads(4);
#endif

    return mnist();
}
