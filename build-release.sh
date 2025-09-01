#!/usr/bin/env bash
set -e

if [ ! -d bin/release ]; then
    mkdir bin/release
fi

if [ ! -d .obj ]; then
    mkdir .obj
fi

gcc -Iinc -Isrc -c src/network.c -o .obj/network.o
gcc -Iinc -Isrc -c src/dataloader.c -o .obj/dataloader.o
gcc -Iinc -Isrc -c src/sgd.c -o .obj/sgd.o
gcc -Iinc -Isrc -c src/perceptron.c -o .obj/perceptron.o
gcc -Iinc -Isrc -c src/main.c -o .obj/main.o
gcc .obj/*.o -o bin/release/main -lm 
rm .obj/main.o
