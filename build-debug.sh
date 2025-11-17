#!/usr/bin/env bash
set -e

if [ ! -d bin/debug ]; then
    mkdir -p bin/debug
fi

if [ ! -d .obj ]; then
    mkdir .obj
fi

gcc -Og -g -fsanitize=address -fopenmp -Iinc -Isrc -c src/profiler.c -o .obj/profiler.o
gcc -Og -g -fsanitize=address -fopenmp -Iinc -Isrc -c src/dataloader.c -o .obj/dataloader.o
gcc -Og -g -fsanitize=address -fopenmp -Iinc -Isrc -c src/mlp.c -o .obj/mlp.o
gcc -Og -g -fsanitize=address -fopenmp -Iinc -Isrc -c src/main.c -o .obj/main.o
gcc -g -fsanitize=address -fopenmp .obj/*.o -o bin/debug/main -lm
rm .obj/main.o
