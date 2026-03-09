#! /usr/bin/env bash

export OMP_PROC_BIND=close
export OMP_PLACES=threads

echo $OMP_PROC_BIND
echo $OMP_PLACES

echo "8 threds"
export OMP_NUM_THREADS=8
./bin/release/inter-sample-vectorized > ./results/thread-pinning/8.csv

echo "7 threds"
export OMP_NUM_THREADS=7
./bin/release/inter-sample-vectorized > ./results/thread-pinning/7.csv

echo "6 threds"
export OMP_NUM_THREADS=6
./bin/release/inter-sample-vectorized > ./results/thread-pinning/6.csv

echo "5 threds"
export OMP_NUM_THREADS=5
./bin/release/inter-sample-vectorized > ./results/thread-pinning/5.csv

echo "4 threds"
export OMP_NUM_THREADS=4
./bin/release/inter-sample-vectorized > ./results/thread-pinning/4.csv

echo "3 threds"
export OMP_NUM_THREADS=3
./bin/release/inter-sample-vectorized > ./results/thread-pinning/3.csv

echo "2 threds"
export OMP_NUM_THREADS=2
./bin/release/inter-sample-vectorized > ./results/thread-pinning/2.csv

