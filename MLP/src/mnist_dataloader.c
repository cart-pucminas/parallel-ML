#include "mnist_dataloader.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LABEL_MAGIC 2049
#define IMAGE_MAGIC 2051

static const char *systemErrorMessage = NULL;
static const char *internalErrorMessage = NULL;
static const char *errorMessage = NULL;

static int freadBigEndian(void *restrict ptr, size_t size, size_t n,
                          FILE *restrict stream)
{
    if (size == 0 || n == 0)
        return 0;

    size_t readBytes = 0;
    uint8_t *byte = ((uint8_t *)ptr) + size - 1;

    for (size_t i = 0; i < n; i++)
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

    int magic = 1;

    if (freadBigEndian(&magic, sizeof(int), 1, file) != 1 ||
        freadBigEndian(size_p, sizeof(int), 1, file) != 1)
    {
        internalErrorMessage = "Failed to read file";
        fclose(file);
        return 0;
    }

    if (magic != LABEL_MAGIC)
    {
        internalErrorMessage = "Magic number mismatch (expected 2049)";
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
        free(l);
        return 0;
    }

    *labels = l;

    fclose(file);

    return 1;
}

static int readImages(uint8_t ***images, const char *restrict path,
                      size_t *restrict size_p, size_t *restrict width_p,
                      size_t *restrict height_p)
{
    char fullPath[1024];
    snprintf(fullPath, sizeof(fullPath), "%s/%s", ROOT_DIR, path);

    FILE *file = fopen(fullPath, "rb");

    if (file == NULL)
    {
        systemErrorMessage = strerror(errno);
        internalErrorMessage = "Failed to open labels file";
        return 0;
    }

    int magic = 1;

    if (freadBigEndian(&magic, sizeof(int), 1, file) != 1 ||
        freadBigEndian(size_p, sizeof(int), 1, file) != 1 ||
        freadBigEndian(height_p, sizeof(int), 1, file) != 1 ||
        freadBigEndian(width_p, sizeof(int), 1, file) != 1)
    {
        internalErrorMessage = "Failed to read file";
        fclose(file);
        return 0;
    }

    if (magic != IMAGE_MAGIC)
    {
        internalErrorMessage = "Magic number mismatch (expected 2051)";
        return 0;
    }

    uint8_t **imgs = malloc(*size_p * sizeof(uint8_t *));

    if (imgs == NULL)
    {
        systemErrorMessage = strerror(errno);
        internalErrorMessage = "Memory allocation error";
        fclose(file);
        return 0;
    }

    size_t pixelCount = (*width_p) * (*height_p);

    for (size_t i = 0; i < *size_p; i++)
    {
        imgs[i] = malloc(pixelCount * sizeof(uint8_t));

        if (imgs[i] == NULL)
        {
            systemErrorMessage = strerror(errno);
            internalErrorMessage = "Memory allocation error";
            fclose(file);
            for (size_t j = 0; j < i; j++)
                free(imgs[j]);
            free(imgs);
            return 0;
        }

        if (fread(imgs[i], sizeof(uint8_t), pixelCount, file) != pixelCount)
        {
            internalErrorMessage = "Failed to read file";
            fclose(file);
            for (size_t j = 0; j < i; j++)
                free(imgs[j]);
            free(imgs);
            return 0;
        }
    }

    *images = imgs;

    fclose(file);

    return 1;
}

int mnist_loadDataset(Dataset **dataset, const char *restrict labelsPath,
                      const char *restrict imagesPath)
{
    size_t labelsSize = 0, imagesSize = 0, imageWidth = 0, imageHeight = 0;

    uint8_t *labels = NULL;
    if (!readLabels(&labels, labelsPath, &labelsSize))
    {
        errorMessage = "Could not read labels";
        return 0;
    }

    uint8_t **images = NULL;
    if (!readImages(&images, imagesPath, &imagesSize, &imageWidth,
                    &imageHeight))
    {
        errorMessage = "Could not read images";
        free(labels);
        return 0;
    }

    if (labelsSize != imagesSize)
    {
        errorMessage = "Label and image count are not the same";
        free(labels);
        for (size_t i = 0; i < imagesSize; i++)
            free(images[i]);
        free(images);
        return 0;
    }

    Dataset *d = malloc(sizeof(Dataset));
    d->size = labelsSize;

    d->inputs = malloc(labelsSize * sizeof(float *));
    d->groundTruths = malloc(labelsSize * sizeof(float *));

    for (unsigned int i = 0; i < labelsSize; i++)
    {
        d->inputs[i] = malloc(imageWidth * imageHeight * sizeof(float));

        for (unsigned int j = 0; j < imageWidth * imageHeight; j++)
            d->inputs[i][j] = images[i][j] / 255.0f;

        d->groundTruths[i] = malloc(10 * sizeof(float));

        for (int j = 0; j < 10; j++)
            d->groundTruths[i][j] = (float)(j == labels[i]);
    }

    *dataset = d;

    free(labels);
    for (size_t i = 0; i < imagesSize; i++)
        free(images[i]);
    free(images);

    return 1;
}

const char *mnist_dataset_getError()
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
