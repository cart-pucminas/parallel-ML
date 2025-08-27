#!/usr/bin/env bash
set -e

if [ ! -d bin/profiling/feed-forward ]; then
    mkdir bin/profiling/feed-forward
fi

if [ ! -d .obj ]; then
    mkdir .obj
fi

./build-profile.sh profile-ff profiling/feed-forward/p0 
./build-profile.sh profile-ff profiling/feed-forward/p1 "-DFEED_FORWARD_PARALLEL_1" 
./build-profile.sh profile-ff profiling/feed-forward/p2 "-DFEED_FORWARD_PARALLEL_2"
