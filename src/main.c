#include <stdio.h>
#include <string.h>

#include "dataloader.h"
#include "perceptron.h"

const char grayscaleMap[] = ".:-=+*#%@";

int main(int argc, char **argv)
{
    char *labels =
             "/home/mateus/repos/perceptron/input/train-labels.idx1-ubyte",
         *images =
             "/home/mateus/repos/perceptron/input/train-images.idx3-ubyte";
    int mode = -1;
    ActivationFunction activation = SIGMOID;

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

    learn(learningDataset);
    classify(classificationDataset);

    printf("all done :D\n");
    return 0;
}

int parseArgs(int argc, char **argv, char *labels, char *images, int mode,
              ActivationFunction activation)
{
    for (int i = 1; i < argc; i++)
    {
        int size = strlen(argv[i]);

        if (size < 2 || argv[i][0] != '-' || (size > 2 && argv[i][1] != '-'))
        {
            printf("unknown argument %s\n", argv[i]);
            return 1;
        }

        if (strncmp(argv[i], "-i", size) == 0 ||
            strncmp(argv[i], "--images", size) == 0)
            images = argv[++i];
        else if (strncmp(argv[i], "-l", size) == 0 ||
                 strncmp(argv[i], "--labels", size) == 0)
            labels = argv[++i];
        else if (strncmp(argv[i], "-t", size) == 0 ||
                 strncmp(argv[i], "--train", size) == 0)
        {
            if (mode == -1)
                mode = 0;
            else
            {
                printf("execution mode already set");
                return 1;
            }
        }
        else if (strncmp(argv[i], "-c", size) == 0 ||
                 strncmp(argv[i], "--classify", size) == 0)
        {
            if (mode == -1)
                mode = 1;
            else
            {
                printf("execution mode already set");
                return 1;
            }
        }
        else if (strncmp(argv[i], "--activation", 12) == 0)
        {
            if (size <= 12 || argv[i][12] != '=')
            {
                printf("'--activation' requires a value\n");
                return 1;
            }
            char *value = argv[i] + 13;
            size = strlen(value);
            if (strncmp(value, "sigmoid", size) == 0)
                activation = SIGMOID;
            else if (strncmp(value, "tanh", size) == 0)
                activation = TANH;
            else if (strncmp(value, "relu", size) == 0)
                activation = RELU;
            else if (strncmp(value, "softmax", size) == 0)
                activation = SOFTMAX;
            else
            {
                printf("unknown --activation value\n");
                return 1;
            }
        }
        else if (strncmp(argv[i], "--load", size) == 0)
        {
        }
    }

    if (mode < 0)
    {
        printf("execution mode not set\n");
        return 1;
    }

    if (labels == NULL)
    {
        printf("labels path not set\n");
        return 1;
    }

    if (images == NULL)
    {
        printf("images path not set\n");
        return 1;
    }
}
