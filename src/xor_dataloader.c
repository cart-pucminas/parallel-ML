#ifndef ROOT_DIR
#define ROOT_DIR "."
#endif

#include "xor_dataloader.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LABEL_MAGIC 2049
#define IMAGE_MAGIC 2051

const char *systemErrorMessage = NULL;
const char *internalErrorMessage = NULL;
const char *errorMessage = NULL;

uint8_t *readLabels(const char *restrict path, size_t *restrict size_p)
{
    char fullPath[1024];
    (void)snprintf(fullPath, sizeof(fullPath), "%s/%s", ROOT_DIR, path);

    FILE *file = fopen(fullPath, "rb");

    if (file == NULL)
    {
        systemErrorMessage = strerror(errno);
        internalErrorMessage = "Failed to open file";
        return NULL;
    }

    if (fread(size_p, sizeof(int), 1, file) != 1)
    {
        internalErrorMessage = "Failed to read file";
        fclose(file);
        return NULL;
    }

    uint8_t *labels = malloc(*size_p * sizeof(uint8_t));

    if (labels == NULL)
    {
        systemErrorMessage = strerror(errno);
        internalErrorMessage = "Memory allocation error";
        fclose(file);
        return NULL;
    }

    if (fread(labels, sizeof(uint8_t), *size_p, file) != *size_p)
    {
        internalErrorMessage = "Failed to read file";
        fclose(file);
        free(labels);
        return NULL;
    }

    fclose(file);

    return labels;
}

uint8_t **readInputs(const char *restrict path, size_t *restrict size_p)
{
    char fullPath[1024];
    snprintf(fullPath, sizeof(fullPath), "%s/%s", ROOT_DIR, path);

    FILE *file = fopen(fullPath, "rb");

    if (file == NULL)
    {
        systemErrorMessage = strerror(errno);
        internalErrorMessage = "Failed to open inputs file";
        return NULL;
    }

    int magic = 1;

    if (fread(size_p, sizeof(int), 1, file) != 1)
    {
        internalErrorMessage = "Failed to read file";
        fclose(file);
        return NULL;
    }

    uint8_t **images = malloc(*size_p * sizeof(uint8_t *));

    if (images == NULL)
    {
        systemErrorMessage = strerror(errno);
        internalErrorMessage = "Memory allocation error";
        fclose(file);
        return NULL;
    }

    for (size_t i = 0; i < *size_p; i++)
    {
        images[i] = malloc(2 * sizeof(uint8_t));

        if (images[i] == NULL)
        {
            systemErrorMessage = strerror(errno);
            internalErrorMessage = "Memory allocation error";
            fclose(file);
            for (size_t j = 0; j < i; j++)
                free(images[j]);
            free(images);
            return NULL;
        }

        if (fread(images[i], sizeof(uint8_t), 2, file) != 2)
        {
            internalErrorMessage = "Failed to read file";
            fclose(file);
            for (size_t j = 0; j < i; j++)
                free(images[j]);
            free(images);
            return NULL;
        }
    }

    fclose(file);

    return images;
}

XorDataset *loadDataset(const char *restrict labelsPath,
                        const char *restrict imagesPath)
{
    size_t labelsSize = 0, inputsSize = 0;

    uint8_t *labels = readLabels(labelsPath, &labelsSize);
    if (labels == NULL)
    {
        errorMessage = "Could not read labels";
        return NULL;
    }

    uint8_t **inputs = readInputs(imagesPath, &inputsSize);
    if (inputs == NULL)
    {
        errorMessage = "Could not read inputs";
        return NULL;
        free(labels);
    }

    if (labelsSize != inputsSize)
    {
        errorMessage = "Label and input count are not the same";
        return NULL;
        free(labels);
        for (size_t i = 0; i < inputsSize; i++)
            free(inputs[i]);
        free(inputs);
    }

    XorDataset *dataset = malloc(sizeof(XorDataset));
    dataset->labels = labels;
    dataset->inputs = inputs;
    dataset->size = labelsSize;

    return dataset;
}

int freeDataset(XorDataset *dataset)
{
    for (size_t i = 0; i < dataset->size; i++)
    {
        free(dataset->inputs[i]);
    }

    free(dataset->inputs);
    free(dataset->labels);
    free(dataset);

    return 1;
}

const char *dataset_getError()
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
