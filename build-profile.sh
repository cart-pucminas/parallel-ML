#!/usr/bin/env bash 

PROFILING_NAME=$1
SGD_FULL_PATH=$2
BIN_FULL_PATH=$3

SGD_NAME=$(basename "$SGD_FULL_PATH" .c)

gcc -w -fopenmp -Iinc -Isrc -c src/network.c -o .obj/network.o
gcc -w -fopenmp -Iinc -Isrc -c src/dataloader.c -o .obj/dataloader.o
gcc -w -fopenmp -Iinc -Isrc -c $SGD_FULL_PATH -o .obj/$SGD_NAME.o
gcc -w -fopenmp -Iinc -Isrc -c src/perceptron.c -o .obj/perceptron.o
gcc -w -fopenmp -Iinc -Isrc -c profiling/profiler.c -o .obj/profiler.o
gcc -w -fopenmp -Iinc -Isrc -c profiling/$PROFILING_NAME.c -o .obj/$PROFILING_NAME.o
gcc -w .obj/*.o -o bin/$BIN_FULL_PATH -lm -fopenmp
rm .obj/*.o

echo "profile available at bin/$BIN_FULL_PATH"
