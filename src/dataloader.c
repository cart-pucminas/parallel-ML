#include "dataloader.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dataset.h"

#define LABEL_MAGIC 2049
#define IMAGE_MAGIC 2051

const char *systemErrorMessage = NULL;
const char *internalErrorMessage = NULL;
const char *errorMessage = NULL;

int freadBigEndian(void *restrict ptr, size_t size, size_t n,
                   FILE *restrict stream)
{
    if (size == 0 || n == 0)
        return 0;

    size_t readBytes = 0;
    uint8_t *byte = ((uint8_t *)ptr) + size - 1;

    for (int i = 0; i < n; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            if (fread(byte, sizeof(uint8_t), 1, stream) != 1)
            {
                systemErrorMessage = strerror(errno);
                return readBytes / size;
            }

            byte--;
            readBytes++;
        }
        byte += (size * 2) - 1;
    }

    return readBytes / size;
}

uint8_t *readLabels(const char *restrict path, int *restrict size_p)
{
    FILE *file = fopen(path, "rb");

    if (file == NULL)
    {
        systemErrorMessage = strerror(errno);
        internalErrorMessage = "Failed to open file";
        return NULL;
    }

    int magic = 1;

    if (freadBigEndian(&magic, sizeof(int), 1, file) != 1 ||
        freadBigEndian(size_p, sizeof(int), 1, file) != 1)
    {
        internalErrorMessage = "Failed to read file";
        fclose(file);
        return NULL;
    }

    if (magic != LABEL_MAGIC)
    {
        internalErrorMessage = "Magic number mismatch (expected 2049)";
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

    return labels;
}

uint8_t **readImages(const char *restrict path, int *restrict size_p,
                     int *restrict width_p, int *restrict height_p)
{
    FILE *file = fopen(path, "rb");

    if (file == NULL)
    {
        systemErrorMessage = strerror(errno);
        internalErrorMessage = "Failed to open labels file";
        return NULL;
    }

    int magic = 1;

    if (freadBigEndian(&magic, sizeof(int), 1, file) != 1 ||
        freadBigEndian(size_p, sizeof(int), 1, file) != 1 ||
        freadBigEndian(height_p, sizeof(int), 1, file) != 1 ||
        freadBigEndian(width_p, sizeof(int), 1, file) != 1)
    {
        internalErrorMessage = "Failed to read file";
        fclose(file);
        return NULL;
    }

    if (magic != IMAGE_MAGIC)
    {
        internalErrorMessage = "Magic number mismatch (expected 2051)";
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

    int pixelCount = (*width_p) * (*height_p);

    for (int i = 0; i < *size_p; i++)
    {
        images[i] = malloc(pixelCount * sizeof(uint8_t));

        if (images[i] == NULL)
        {
            systemErrorMessage = strerror(errno);
            internalErrorMessage = "Memory allocation error";
            fclose(file);
            for (int j = 0; j < i; j++)
                free(images[j]);
            free(images);
            return NULL;
        }

        if (fread(images[i], sizeof(uint8_t), pixelCount, file) != pixelCount)
        {
            internalErrorMessage = "Failed to read file";
            fclose(file);
            for (int j = 0; j < i; j++)
                free(images[j]);
            free(images);
            return NULL;
        }
    }

    fclose(file);
    return images;
}

Dataset *loadDataset(const char *restrict labelsPath,
                     const char *restrict imagesPath)
{
    int labelsSize = 0, imagesSize = 0, imageWidth = 0, imageHeight = 0;

    uint8_t *labels = readLabels(labelsPath, &labelsSize);
    if (labels == NULL)
    {
        errorMessage = "Could not read labels";
        return NULL;
    }

    uint8_t **images =
        readImages(imagesPath, &imagesSize, &imageWidth, &imageHeight);
    if (images == NULL)
    {
        errorMessage = "Could not read images";
        return NULL;
    }

    if (labelsSize != imagesSize)
    {
        errorMessage = "Label and image count are not the same";
        return NULL;
    }

    Dataset *dataset = malloc(sizeof(Dataset));
    dataset->labels = labels;
    dataset->images = images;
    dataset->size = labelsSize;
    dataset->imageWidth = imageWidth;
    dataset->imageHeight = imageHeight;

    return dataset;
}

int closeDataset(Dataset *dataset)
{
    for (int i = 0; i < dataset->size; i++)
    {
        free(dataset->images[i]);
    }

    free(dataset->images);
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
        snprintf(message, sizeof(message), "%s: %s", message,
                 internalErrorMessage);
    if (systemErrorMessage)
        snprintf(message, sizeof(message), "%s (%s)", message,
                 systemErrorMessage);

    return message;
}
