#ifndef DATASET_H
#define DATASET_H

#include <stdlib.h>

typedef struct
{
    float **inputs;
    float **groundTruths;
    size_t size;
} Dataset;

#endif
