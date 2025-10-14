#!/usr/bin/env bash
set -e

if [ ! -d bin/profiling/feed-forward ]; then
    mkdir bin/profiling/feed-forward
fi

if [ ! -d .obj ]; then
    mkdir .obj
fi

./build-profile.sh profile-ff src/sgd.c profiling/feed-forward/single
./build-profile.sh profile-ff profiling/sgd/sgd-ff1-v1.c profiling/feed-forward/p1-v1 
./build-profile.sh profile-ff profiling/sgd/sgd-ff1-v2.c profiling/feed-forward/p2-v1 
./build-profile.sh profile-ff profiling/sgd/sgd-ff2-v1.c profiling/feed-forward/p1-v2 
./build-profile.sh profile-ff profiling/sgd/sgd-ff2-v2.c profiling/feed-forward/p2-v2 
./build-profile.sh profile-ff profiling/sgd/sgd-ff3-v1.c profiling/feed-forward/p3-v1 
./build-profile.sh profile-ff profiling/sgd/sgd-ff3-v2.c profiling/feed-forward/p3-v2 
