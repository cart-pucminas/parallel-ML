#ifndef XOR_DATALOADER_H
#define XOR_DATALOADER_H

#include "mlp.h"

int xor_loadDataset(Dataset **dataset, const char *labelPath,
                    const char *inputPath);

const char *xor_dataset_getError(void);

#endif
