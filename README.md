# Learning Parallel Computing with Multilayer Perceptron Neural Networks

- Author: Mateus Henrique Medeiros Diniz (mateushmdiniz@gmail.com)
- Advisor: Henrique Cota de Freitas (cota@pucminas.com)

A Multilayer Perceptron made from scratch in C with the goal of tackling hard parallel programming concepts with a highly parallelizable example.

Used the MNIST Dataset. Available at [http://yann.lecun.com/exdb/mnist/].

## Parallelism Strategies

This implementation supports multiple parallelization strategies toggleable at compile-time:
* **Inter-sample:** Parallelizes the mini-batch processing using OpenMP reductions.
* **Intra-layer:** Parallelizes the neuron activations and weight updates within layers.
* **SIMD Optimized:** Utilizes vectorized instructions for core math routines.

## Requirements

- GCC (version 14.3.0 for maximum stability)
- Make

## Building

Inside the root of the project:

```
make
./bin/release/...
```

Other compilation options can be seen in `Makefile`.
