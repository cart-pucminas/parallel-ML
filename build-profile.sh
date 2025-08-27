#!/usr/bin/env bash 

NAME=$1
BINFULLPATH=$2
FLAGS=$3

gcc $FLAGS -w -fopenmp -Iinc -Isrc -c src/profiler.c -o .obj/profiler.o
gcc $FLAGS -w -fopenmp -Iinc -Isrc -c src/network.c -o .obj/network.o
gcc $FLAGS -w -fopenmp -Iinc -Isrc -c src/dataloader.c -o .obj/dataloader.o
gcc $FLAGS -w -fopenmp -Iinc -Isrc -c src/sgd.c -o .obj/sgd.o
gcc $FLAGS -w -fopenmp -Iinc -Isrc -c src/perceptron.c -o .obj/perceptron.o

gcc $FLAGS -w -fopenmp -Iinc -Isrc -c profiling/$NAME.c -o .obj/$NAME.o
gcc -w .obj/*.o -o bin/$BINFULLPATH -lm -fopenmp
rm .obj/*.o

echo "profile available at bin/$BINFULLPATH"
