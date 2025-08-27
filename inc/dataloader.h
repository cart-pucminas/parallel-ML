#ifndef DATALOADER_H
#define DATALOADER_H

#include "dataset.h"

Dataset *loadDataset(const char *labelPath, const char *imagesPath);

int freeDataset(Dataset *dataset);

const char *dataset_getError(void);

#endif
