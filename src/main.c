#ifndef ROOT_DIR
#define ROOT_DIR "."
#endif

#include <omp.h>
#include <stdio.h>

#include "mlp.h"
#include "mnist_dataloader.h"
#include "xor_dataloader.h"

const char grayscaleMap[] = ".:-=+*#%@";

int main()
{
    omp_set_num_threads(4);
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
    Network *n = constructNetwork(1, 3, layers, 0.5, 100);

    fit(n, learningDataset);
    classify(n, classificationDataset);

    freeDataset(learningDataset);
    freeDataset(classificationDataset);
    freeNetwork(n);

    printf("all done :D\n");
    return 0;
}
