
#include "xor_dataloader.h"
#include "mlp.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *systemErrorMessage = NULL;
static const char *internalErrorMessage = NULL;
static const char *errorMessage = NULL;

static int readLabels(uint8_t **labels, const char *restrict path,
                      size_t *restrict size_p)
{
    char fullPath[1024];
    (void)snprintf(fullPath, sizeof(fullPath), "%s/%s", ROOT_DIR, path);

    FILE *file = fopen(fullPath, "rb");

    if (file == NULL)
    {
        systemErrorMessage = strerror(errno);
        internalErrorMessage = "Failed to open file";
        return 0;
    }

    if (fread(size_p, sizeof(int), 1, file) != 1)
    {
        internalErrorMessage = "Failed to read file";
        fclose(file);
        return 0;
    }

    uint8_t *l = malloc(*size_p * sizeof(uint8_t));

    if (l == NULL)
    {
        systemErrorMessage = strerror(errno);
        internalErrorMessage = "Memory allocation error";
        fclose(file);
        return 0;
    }

    if (fread(l, sizeof(uint8_t), *size_p, file) != *size_p)
    {
        internalErrorMessage = "Failed to read file";
        fclose(file);
        free(labels);
        return 0;
    }

    *labels = l;

    fclose(file);

    return 1;
}

static int readInputs(uint8_t ***inputs, const char *restrict path,
                      size_t *restrict size_p)
{
    char fullPath[1024];
    snprintf(fullPath, sizeof(fullPath), "%s/%s", ROOT_DIR, path);

    FILE *file = fopen(fullPath, "rb");

    if (file == NULL)
    {
        systemErrorMessage = strerror(errno);
        internalErrorMessage = "Failed to open inputs file";
        return 0;
    }

    if (fread(size_p, sizeof(int), 1, file) != 1)
    {
        internalErrorMessage = "Failed to read file";
        fclose(file);
        return 0;
    }

    uint8_t **in = malloc(*size_p * sizeof(uint8_t *));

    if (in == NULL)
    {
        systemErrorMessage = strerror(errno);
        internalErrorMessage = "Memory allocation error";
        fclose(file);
        return 0;
    }

    for (size_t i = 0; i < *size_p; i++)
    {
        in[i] = malloc(2 * sizeof(uint8_t));

        if (in[i] == NULL)
        {
            systemErrorMessage = strerror(errno);
            internalErrorMessage = "Memory allocation error";
            fclose(file);
            for (size_t j = 0; j < i; j++)
                free(in[j]);
            free(in);
            return 0;
        }

        if (fread(in[i], sizeof(uint8_t), 2, file) != 2)
        {
            internalErrorMessage = "Failed to read file";
            fclose(file);
            for (size_t j = 0; j < i; j++)
                free(in[j]);
            free(in);
            return 0;
        }
    }

    *inputs = in;

    fclose(file);

    return 1;
}

int xor_loadDataset(Dataset **dataset, const char *restrict labelsPath,
                    const char *restrict inputsPath)
{
    size_t labelsSize = 0, inputsSize = 0;

    uint8_t *labels = NULL;

    if (!readLabels(&labels, labelsPath, &labelsSize))
    {
        errorMessage = "Could not read labels";
        return 0;
    }

    uint8_t **inputs = NULL;

    if (!readInputs(&inputs, inputsPath, &inputsSize))
    {
        errorMessage = "Could not read inputs";
        free(labels);
        return 0;
    }

    if (labelsSize != inputsSize)
    {
        errorMessage = "Label and input count are not the same";
        free(labels);
        for (size_t i = 0; i < inputsSize; i++)
            free(inputs[i]);
        free(inputs);
        return 0;
    }

    Dataset *d = malloc(sizeof(Dataset));

    d->size = labelsSize;

    d->groundTruths = malloc(labelsSize * sizeof(float *));
    d->inputs = malloc(labelsSize * sizeof(float *));

    for (size_t i = 0; i < labelsSize; i++)
    {
        d->groundTruths[i] = malloc(1 * sizeof(float));
        d->groundTruths[i][0] = (float)labels[0];

        d->inputs[i] = malloc(2 * sizeof(float));
        d->inputs[i][0] = inputs[i][0];
        d->inputs[i][1] = inputs[i][1];
    }

    *dataset = d;

    free(labels);
    for (size_t i = 0; i < inputsSize; i++)
        free(inputs[i]);
    free(inputs);

    return 1;
}

const char *xor_dataset_getError()
{
    static char message[512];

    if (errorMessage)
        snprintf(message, sizeof(message), "%s", errorMessage);
    if (internalErrorMessage)
    {
        size_t len = strlen(message);
        snprintf(message + len, sizeof(message) - len, ": %s",
                 internalErrorMessage);
    }
    if (systemErrorMessage)
    {
        size_t len = strlen(message);
        snprintf(message + len, sizeof(message) - len, " (%s)",
                 systemErrorMessage);
    }

    return message;
}
