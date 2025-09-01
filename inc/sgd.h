#ifndef SGD_H
#define SGD_H

#include "network.h"

#define STRATEGY_0
#define STRATEGY_1 _Pragma("omp parallel for")
#define STRATEGY_2 _Pragma("omp parallel for schedule(dynamic)")

#ifndef FFP_1_V1
#define FFP_1_V1 0
#endif

#ifndef FFP_1_V2
#define FFP_1_V2 0
#endif

#ifndef BPP_1_V1
#define BPP_1_V1 0
#endif

#ifndef BPP_1_1_V1
#define BPP_1_1_V1 0
#endif

#ifndef BPP_1_1_V2
#define BPP_1_1_V2 0
#endif

#ifndef BPP_1_2_V1
#define BPP_1_2_V1 0
#endif

#ifndef BPP_1_2_V2
#define BPP_1_2_V2 0
#endif

#ifndef BPP_1_2_V3
#define BPP_1_2_V3 0
#endif

#ifndef BPP_1_2_3_V1
#define BPP_1_2_3_V1 0
#endif

#if FFP_1_V1 && FFP_1_V2
#error "Conflicting Feed Forward parallel strategies"
#endif

#if BPP_1_V1 && (BPP_1_1_V1 || BPP_1_1_V2 || BPP_1_2_V1 || BPP_1_2_V2 ||            \
                BPP_1_2_V3 || BPP_1_2_3_V1)
#error "Conflicting Backpropagation parallel strategies"
#endif

#if BPP_1_1_V1 && BPP_1_1_V2
#error "Conflicting Backpropagation parallel strategies"
#endif

#if BPP_1_2_V1 && BPP_1_2_V2 || BPP_1_2_V1 && BPP_1_2_V3 || BPP_1_2_V2 && BPP_1_2_V3
#error "Conflicting Backpropagation parallel strategies"
#endif

#if FFP_1_V1 == 2
#define FEED_FORWARD_PARALLEL_STRATEGY_1_1 STRATEGY_2
#elif FFP_1_V1 == 1
#define FEED_FORWARD_PARALLEL_STRATEGY_1_1 STRATEGY_1
#else
#define FEED_FORWARD_PARALLEL_STRATEGY_1_1
#endif

#if FFP_1_V2 == 2
#define FEED_FORWARD_PARALLEL_STRATEGY_1_2 STRATEGY_2
#elif FFP_1_V2 == 1
#define FEED_FORWARD_PARALLEL_STRATEGY_1_2 STRATEGY_1
#else
#define FEED_FORWARD_PARALLEL_STRATEGY_1_2
#endif

#if BPP_1_V1 == 2
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_1 STRATEGY_2
#elif BPP_1_V1 == 1
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_1 STRATEGY_1
#else
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_1
#endif

#if BPP_1_1_V1 == 2
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_1_1 STRATEGY_2
#elif BPP_1_1_V1 == 1
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_1_1 STRATEGY_1
#else
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_1_1
#endif

#if BPP_1_1_V2 == 2
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_1_2 STRATEGY_2
#elif BPP_1_1_V2 == 1
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_1_2 STRATEGY_1
#else
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_1_2
#endif

#if BPP_1_2_V1 == 2
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_1 STRATEGY_2
#elif BPP_1_2_V1 == 1
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_1 STRATEGY_1
#else
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_1
#endif

#if BPP_1_2_V2 == 2
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_2 STRATEGY_2
#elif BPP_1_2_V2 == 1
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_2 STRATEGY_1
#else
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_2
#endif

#if BPP_1_2_V3 == 2
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_3 STRATEGY_2
#elif BPP_1_2_V3 == 1
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_3 STRATEGY_1
#else
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_3
#endif

#if BPP_1_2_3_V1 == 2
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_3_1 STRATEGY_2
#elif BPP_1_2_3_V1 == 1
#define BACK_PROPAGATION_PARALLEL_STRATEGY_1_2_3_1 STRATEGY_1
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
