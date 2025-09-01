#!/usr/bin/env bash
set -e

if [ ! -d bin/profiling/feed-forward ]; then
    mkdir bin/profiling/feed-forward
fi

if [ ! -d .obj ]; then
    mkdir .obj
fi

./build-profile.sh profile-ff profiling/feed-forward/p0 
./build-profile.sh profile-ff profiling/feed-forward/p1-v1 "-DFFP_1_1=1" 
./build-profile.sh profile-ff profiling/feed-forward/p2-v1 "-DFFP_1_2=1"
./build-profile.sh profile-ff profiling/feed-forward/p1-v2 "-DFFP_1_1=2" 
./build-profile.sh profile-ff profiling/feed-forward/p2-v2 "-DFFP_1_2=2"

