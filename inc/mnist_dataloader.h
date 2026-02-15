#ifndef DATALOADER_H
#define DATALOADER_H

#include <stdint.h>
#include <stdlib.h>

typedef struct
{
    uint8_t *labels;
    uint8_t **images;
    size_t size;
    size_t imageWidth;
    size_t imageHeight;
} MnistDataset;

MnistDataset *loadDataset(const char *labelPath, const char *imagesPath);

int freeDataset(MnistDataset *dataset);

const char *dataset_getError(void);

#endif
