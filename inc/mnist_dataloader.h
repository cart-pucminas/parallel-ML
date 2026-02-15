#ifndef MNIST_DATALOADER_H
#define MNIST_DATALOADER_H

#include <mlp.h>

int mnist_loadDataset(Dataset **dataset, const char *labelPath,
                      const char *imagesPath);

const char *mnist_dataset_getError(void);

#endif
