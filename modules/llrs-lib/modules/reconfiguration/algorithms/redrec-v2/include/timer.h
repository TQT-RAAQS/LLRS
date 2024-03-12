#ifndef _TIMER_H_
#define _TIMER_H_

#include <stdio.h>
#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

static void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

static void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

// time is returned in nanoseconds

static void printElapsedTime(Timer timer, const char* s) {
    float t = ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec)*1.0e9 \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)*1.0e3));
    printf("%s: %f s\n", s, t);
}

static float getElapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec)*1.0e9 \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)*1.0e3));
}

#endif

