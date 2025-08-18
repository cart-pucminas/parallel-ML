#!/usr/bin/env bash
set -e

if [ ! -d bin/debug ]; then
    mkdir bin/debug
fi

if [ ! -d .obj ]; then
    mkdir .obj
fi

gcc -O0 -g -fsanitize=address -Iinc -Isrc -c src/network.c -o .obj/network.o
gcc -O0 -g -fsanitize=address -Iinc -Isrc -c src/dataloader.c -o .obj/dataloader.o
gcc -O0 -g -fsanitize=address -Iinc -Isrc -c src/sgd.c -o .obj/sgd.o
gcc -O0 -g -fsanitize=address -Iinc -Isrc -c src/perceptron.c -o .obj/perceptron.o
gcc -O0 -g -fsanitize=address -Iinc -Isrc -c src/main.c -o .obj/main.o
gcc -g -fsanitize=address .obj/*.o -o bin/debug/main -lm
