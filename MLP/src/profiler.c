#include "profiler.h"

#include <time.h>

void profile_start(Timer *timer)
{
    clock_gettime(CLOCK_MONOTONIC, &timer->start);
}

double profile_getElapsed(Timer *timer)
{
    struct timespec end;

    clock_gettime(CLOCK_MONOTONIC, &end);

    double seconds = (double)(end.tv_sec - timer->start.tv_sec);
    double nanoseconds = (double)(end.tv_nsec - timer->start.tv_nsec);

    if (nanoseconds < 0)
    {
        seconds--;
        nanoseconds += 1e9;
    }

    return seconds + (nanoseconds / 1e9);
}
