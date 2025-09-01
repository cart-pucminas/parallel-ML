#include "profiler.h"
#include "bits/time.h"

#include <time.h>

static struct timespec startTime;

void profile_start() 
{ 
    clock_gettime(CLOCK_MONOTONIC, &startTime); 
}

double profile_getElapsed()
{
    struct timespec endTime;
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    return (endTime.tv_sec - startTime.tv_sec) +
           (endTime.tv_nsec - startTime.tv_nsec) / 1e9;
}
