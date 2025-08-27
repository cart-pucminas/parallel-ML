#ifndef ARGS_H
#define ARGS_H

#include <stdio.h>
#include <string.h>

#include "perceptron.h"

Params *parseArgs(int argc, char **argv)
{
    Params *params = calloc(1, sizeof(Params));
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
            int *sizes = malloc(100 * sizeof(int));
            int idx = 0;
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
                pTok = strtok(NULL, ",");
                idx++;
            }
            params->hiddenLayerCount = idx;
            params->hiddenLayerSizes = malloc(idx * sizeof(int));
            for (int j = 0; j < idx; j++)
                params->hiddenLayerSizes[j] = sizes[j];
            free(sizes);
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

                if ((value[j] == '.' && dot) ||
                    (value[j] != '.' && (offset < 0 || offset > 9)))
                {
                    printf("'learning-rate' requires a decimal value\n");
                    return NULL;
                }

                if (value[j] == '.')
                    dot = 1;
            }
            char *pEnd;
            params->learningRate = strtof(value, &pEnd);
        }
        else if (strncmp(argv[i], "--batch-size", 12) == 0)
        {
            if (argv[i][12] != '=')
            {
                printf("'--batch-size' requires an integer value\n");
                return NULL;
            }
            char *value = argv[i] + 13;
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
            params->miniBatchSize = strtol(value, &pEnd, 10);
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
            params->epochs = strtol(value, &pEnd, 10);
        }
    }

    return params;
}

#endif
