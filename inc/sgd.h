#ifndef SGD_H
#define SGD_H

#include "network.h"

#if defined(FEED_FORWARD_PARALLEL_1)
#define USE_FF_STRATEGY _Pragma( "omp parallel for" )
#elif defined(FEED_FORWARD_PARALLEL_2)
#define USE_FF_STRATEGY _Pragma( "omp parallel for schedule(dynamic)" )
#else
#define USE_FF_STRATEGY
#endif

typedef enum
{
    SIGMOID,
    TANH,
    RELU,
    SOFTMAX
} ActivationFunction;

void feedForward(NN *network, float **dActZ, ActivationFunction activation);
void sgd_learn(NN *network, Batch *batch, ActivationFunction activation);
void sgd_classify(NN *network, Batch *batch, ActivationFunction activation);

#endif
