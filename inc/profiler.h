#ifndef PROFILER_H
#define PROFILER_H

#include <time.h>

typedef struct
{
    struct timespec start;
} Timer;

void profile_start(Timer *timer);
double profile_getElapsed(Timer *timer);

#endif
