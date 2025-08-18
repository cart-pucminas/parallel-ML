#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "dataset.h"
#include "sgd.h"

void setActivationFunction(ActivationFunction af);
void learn(Dataset *dataset);
void classify(Dataset *dataset);

#endif
