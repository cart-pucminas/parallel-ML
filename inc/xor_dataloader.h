#ifndef DATALOADER_H
#define DATALOADER_H

#include <stdint.h>
#include <stdlib.h>

typedef struct
{
    uint8_t *labels;
    uint8_t **inputs;
    size_t size;
} XorDataset;

XorDataset *loadDataset(const char *labelPath, const char *inputPath);

int freeDataset(XorDataset *dataset);

const char *dataset_getError(void);

#endif
