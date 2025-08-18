#ifndef SGD_H
#define SGD_H

#include "dataset.h"
#include "network.h"

typedef enum
{
    SIGMOID,
    TANH,
    RELU,
    SOFTMAX
} ActivationFunction;

void shuffle(Dataset *dataset);
void sgd_learn(NN *network, Batch *batch, ActivationFunction activation);
void sgd_classify(NN *network, Batch *batch, ActivationFunction activation);

#endif
