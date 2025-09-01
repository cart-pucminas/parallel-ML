#ifndef SGD_H
#define SGD_H

#include "network.h"

#define STRATEGY_0
#define STRATEGY_1 _Pragma("omp parallel for")
#define STRATEGY_2 _Pragma("omp parallel for schedule(dynamic)")

#define DEFINE_STRATEGY (NAME, VALUE)
#define NAME STRATEGY_##VALUE

#if defined(FF_PARALLEL_1_1_1) + defined(FF_PARALLEL_1_1_2) +                  \
        defined(FF_PARALLEL_1_2_1) + defined(FF_PARALLEL_1_2_2) >              \
    1
#error "Multiple Feed Forward parallel strategies defined"
#endif

#if defined(FF_PARALLEL_1_1_1)
#define FEED_FORWARD_PARALLEL_STRATEGY_1_1 _Pragma("omp parallel for")
#elif defined(FF_PARALLEL_1_1_2)
#define FEED_FORWARD_PARALLEL_STRATEGY_1_1                                     \
    _Pragma("omp parallel for schedule(dynamic)")
#else
#define FEED_FORWARD_PARALLEL_STRATEGY_1_1
#endif

#if defined(FF_PARALLEL_1_2_1)
#define FEED_FORWARD_PARALLEL_STRATEGY_1_2 _Pragma("omp parallel for")
#elif defined(FF_PARALLEL_1_2_2)
#define FEED_FORWARD_PARALLEL_STRATEGY_1_2                                     \
    _Pragma("opm parallel for schedule(dynamic)")
#else
#define FEED_FORWARD_PARALLEL_STRATEGY_1_2
#endif

#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_1

#if defined(BP_PARALLEL_1_1_1_1)
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_1_1 _Pragma("omp parallel for")
#elif defined(BP_PARALLEL_1_1_1_2)
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_1_1                               \
    _Pragma("omp parallel for schedule(dynamic)")
#else
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_1_1
#endif

#if defined(BP_PARALLEL_1_1_2_1)
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_1_2 _Pragma("omp parallel for")
#elif defined(BP_PARALLEL_1_1_2_2)
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_1_2                               \
    _Pragma("omp parallel for schedule(dynamic)")
#else
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_1_2
#endif

#if defined(BP_PARALLEL_1_2_1_1)
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_1 _Pragma("omp parallel for")
#elif defined(BP_PARALLEL_1_2_1_2)
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_1                               \
    _Pragma("omp parallel for schedule(dynamic)")
#else
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_1
#endif

#if defined(BP_PARALLEL_1_2_2_1)
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_2 _Pragma("omp parallel for")
#elif defined(BP_PARALLEL_1_2_2_2)
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_2                               \
    _Pragma("omp parallel for schedule(dynamic)")
#else
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_2
#endif

#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_3

#if defined(BP_PARALLEL_1_2_3_1_1)
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_3_1 _Pragma("omp parallel for")
#elif defined(BP_PARALLEL_2_2_1_2)
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_3_1                             \
    _Pragma("omp parallel for schedule(dynamic)")
#else
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_3_1
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
