#pragma once

#include <stdio.h>
#include <vector>

/* Helper functions for debugging.
 */

#ifdef WIN32
/* Trace not ready for linux yet, but just a matter of using the right timing
 * functions */
#include <windows.h>
extern LARGE_INTEGER perfFreq;
#endif

#if defined(ENABLE_TRACE)
#include <vector>

extern std::vector<float> dataValues;
extern std::vector<float> totalValues;
extern std::vector<int> totalCountValues;
extern std::vector<char *> nameValues;
extern CRITICAL_SECTION CriticalSection;
#define SETUP_TRACE                                                            \
    LARGE_INTEGER lastCallCounter2;                                            \
    float currTime;                                                            \
    LARGE_INTEGER newCounter;                                                  \
    QueryPerformanceCounter(&lastCallCounter2);                                \
    if (perfFreq.QuadPart == 0)                                                \
        QueryPerformanceFrequency(&perfFreq);
#define START_TRACE QueryPerformanceCounter(&lastCallCounter2);
#define END_TRACE(var)                                                         \
    QueryPerformanceCounter(&newCounter);                                      \
    currTime = (float(newCounter.QuadPart - lastCallCounter2.QuadPart) /       \
                perfFreq.QuadPart) *                                           \
               1000.0f;                                                        \
    static float var##maxTime = 0;                                             \
    static int var##index = -1;                                                \
    static char *var##name = #var;                                             \
    static int var##startup = 0;                                               \
    var##startup++;                                                            \
    if (var##index == -1) {                                                    \
        EnterCriticalSection(&CriticalSection);                                \
        if (var##index == -1) {                                                \
            var##index = (int)dataValues.size();                               \
            dataValues.push_back(var##maxTime);                                \
            nameValues.push_back(var##name);                                   \
            totalValues.push_back(0);                                          \
            totalCountValues.push_back(0);                                     \
        }                                                                      \
        LeaveCriticalSection(&CriticalSection);                                \
    }                                                                          \
    if (var##startup > 6 && currTime < 200.0) {                                \
        totalValues[var##index] += currTime;                                   \
        totalCountValues[var##index]++;                                        \
        if (currTime > var##maxTime) {                                         \
            var##maxTime = currTime;                                           \
            dataValues[var##index] = var##maxTime;                             \
            if (var##maxTime > 3.0)                                            \
                int p = 1; /*__debugbreak();*/                                 \
        }                                                                      \
    }
#else
#define SETUP_TRACE
#define START_TRACE
#define END_TRACE(var)
#endif

void InitTrace();

void DeInitTrace();
