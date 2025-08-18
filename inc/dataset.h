#ifndef DATASET_H
#define DATASET_H

#include <stdint.h>
#include <stdlib.h>

typedef struct
{
    uint8_t *labels;
    uint8_t **images;
    size_t size;
    size_t imageWidth;
    size_t imageHeight;
} Dataset;

#endif
