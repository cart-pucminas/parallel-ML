#include <stdio.h>

#include "args.h"
#include "network.h"
#include "profiler.h"
#include "sgd.h"


int main(int argc, char **argv)
{
#if defined(FEED_FORWARD_PARALLEL_1)
    printf("static feed forward parallelization\n");
#elif defined (FEED_FORWARD_PARALLEL_2)
    printf("dynamic feed forward parallelization\n");
#else
    printf("single thread feed forward\n");
#endif

    Params *params = parseArgs(argc, argv);

    size_t *sizes = malloc((params->hiddenLayerCount + 2) * sizeof(size_t));

    sizes[0] = 784;
    sizes[params->hiddenLayerCount + 1] = 10;
    for (int i = 0; i < params->hiddenLayerCount; i++)
        sizes[i + 1] = params->hiddenLayerSizes[i];

    NN *network = constructNetwork(params->hiddenLayerCount + 2, sizes,
                                   params->learningRate);

    free(sizes);

    float **dActZ = malloc((network->layerCount - 1) * sizeof(float *));
    for (int i = 1; i < network->layerCount; i++)
        dActZ[i - 1] = malloc(network->layersSizes[i] * sizeof(float));

    profile_start();
    feedForward(network, dActZ, SIGMOID);
    double elapsed = profile_getElapsed();
    printf("%.9f\n", elapsed);

    freeParams(params);
    freeNetwork(network);

    return 0;
}
