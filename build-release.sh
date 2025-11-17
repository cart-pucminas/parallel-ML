#!/usr/bin/env bash
set -e

if [ ! -d bin/release ]; then
    mkdir -p bin/release
fi

if [ ! -d .obj ]; then
    mkdir .obj
fi

gcc -fopenmp -Iinc -Isrc -c src/profiler.c -o .obj/profiler.o
gcc -fopenmp -Iinc -Isrc -c src/dataloader.c -o .obj/dataloader.o
gcc -fopenmp -Iinc -Isrc -c src/mlp.c -o .obj/mlp.o
gcc -fopenmp -Iinc -Isrc -c src/main.c -o .obj/main.o
gcc -fopenmp .obj/*.o -o bin/release/main -lm 
rm .obj/main.o
