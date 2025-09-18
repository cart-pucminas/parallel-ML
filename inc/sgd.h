#ifndef SGD_H
#define SGD_H

#include "network.h"

typedef enum
{
    SIGMOID,
    TANH,
    RELU,
    SOFTMAX
} ActivationFunction;

void sgd_learn(NN *network, Batch *batch, ActivationFunction activation);
void sgd_classify(NN *network, Batch *batch, ActivationFunction activation);

// Exposed for profiling
void feedForward(NN *network, float **dActZ, ActivationFunction activation);
void backPropagation(NN *network, float *groundTruth, float *nablaW,
                     float *nablaB, float **dZActivation, float **partials);

#endif
