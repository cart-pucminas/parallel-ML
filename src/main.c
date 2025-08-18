#include <stdio.h>
#include <string.h>

#include "dataloader.h"
#include "perceptron.h"

const char grayscaleMap[] = ".:-=+*#%@";

Params *parseArgs(int, char **);

int main(int argc, char **argv)
{
    Params *params = parseArgs(argc, argv);

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

Params *parseArgs(int argc, char **argv)
{
    Params *params = malloc(sizeof(Params));
    char *hiddenLayers;

    for (int i = 1; i < argc; i++)
    {
        int size = strlen(argv[i]);

        if (size < 2 || argv[i][0] != '-' || (size > 2 && argv[i][1] != '-'))
        {
            printf("unknown argument %s\n", argv[i]);
            return NULL;
        }

        if (strncmp(argv[i], "--activation", 12) == 0)
        {
            if (argv[i][12] != '=')
            {
                printf("'--activation' requires a value\n");
                return NULL;
            }
            char *value = argv[i] + 13;
            size = strlen(value);
            if (strncmp(value, "sigmoid", size) == 0)
                params->activation = SIGMOID;
            else if (strncmp(value, "tanh", size) == 0)
                params->activation = TANH;
            else if (strncmp(value, "relu", size) == 0)
                params->activation = RELU;
            else if (strncmp(value, "softmax", size) == 0)
                params->activation = SOFTMAX;
            else
            {
                printf("unknown activation value\n");
                return NULL;
            }
        }
        else if (strncmp(argv[i], "--hidden-layers", 15) == 0)
        {
            if (argv[i][15] != '=')
            {
                printf("'--hidden-layers' requires a set of integer values\n");
                return NULL;
            }
            char *value = argv[i] + 16;
            params->hiddenLayerCount = 0;
            int *sizes = malloc(100 * sizeof(int));
            int idx;
            char *pTok = strtok(value, ",");
            while (pTok != NULL)
            {
                size = strlen(pTok);
                for (int j = 0; j < size; j++)
                {
                    int offset = '9' - value[j];

                    if (offset < 0 || offset > 9)
                    {
                        printf("'--hidden-layers' requires a set of integer "
                               "values\n");
                        return NULL;
                    }
                }
                char *pEnd;
                sizes[idx] = strtol(pTok, &pEnd, 10);
                pTok = strtok(value, ",");
            }
        }
        else if (strncmp(argv[i], "--learning-rate", 15) == 0)
        {
            if (argv[i][15] != '=')
            {
                printf("'--learning-rate' requires a decimal value\n");
                return NULL;
            }
            char *value = argv[i] + 16;
            size = strlen(value);
            int dot = 0;
            for (int j = 0; j < size; j++)
            {
                int offset = '9' - value[j];

                if ((value[i] == '.' && dot) || offset < 0 || offset > 9)
                {
                    printf("'--hidden-size' requires an integer value\n");
                    return NULL;
                }

                if (value[i] == '.')
                    dot = 1;
            }
        }
        else if (strncmp(argv[i], "--batch-size", 12) == 0)
        {
            if (argv[i][12] != '=')
            {
                printf("'--batch-size' requires an integer value\n");
                return NULL;
            }
            char *value = argv[i] + 12;
            size = strlen(value);
            for (int j = 0; j < size; j++)
            {
                int offset = '9' - value[j];
                if (offset < 0 || offset > 9)
                {
                    printf("'--batch-size' requires an integer value\n");
                    return NULL;
                }
            }
            char *pEnd;
            params->hiddenLayerCount = strtol(value, &pEnd, 10);
        }
        else if (strncmp(argv[i], "--epochs", 8) == 0)
        {
            if (argv[i][8] != '=')
            {
                printf("'--epochs' requires an integer value\n");
                return NULL;
            }
            char *value = argv[i] + 9;
            size = strlen(value);
            for (int j = 0; j < size; j++)
            {
                int offset = '9' - value[j];
                if (offset < 0 || offset > 9)
                {
                    printf("'--epochs' requires an integer value\n");
                    return NULL;
                }
            }
            char *pEnd;
            params->hiddenLayerCount = strtol(value, &pEnd, 10);
        }
    }

    return params;
}
