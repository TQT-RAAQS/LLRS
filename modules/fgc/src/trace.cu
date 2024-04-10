#include "trace.hpp"
#include <vector>

using namespace std;

// LARGE_INTEGER perfFreq;

#if defined(ENABLE_TRACE)
std::vector<float> dataValues;
std::vector<float> totalValues;
std::vector<int> totalCountValues;
std::vector<char *> nameValues;
#pragma message(__FILE__ ": Trace enabled")
CRITICAL_SECTION CriticalSection;
#else
#define SETUP_TRACE
#endif

void InitTrace() {
#if defined(ENABLE_TRACE)
    if (!InitializeCriticalSectionAndSpinCount(&CriticalSection, 0x00000400))
        __debugbreak();

    for (unsigned int i = 0; i < dataValues.size(); i++) {
        dataValues[i] = 0.0;
        totalValues[i] = 0;
        totalCountValues[i] = 0;
    }
#endif
}

void DeInitTrace() {
#if defined(ENABLE_TRACE)
    for (unsigned int i = 0; i < dataValues.size(); i++) {
        char msg[2048];
        sprintf(msg, "%s MaxTime=%f AverageTime=%f\n", nameValues[i],
                dataValues[i], totalValues[i] / totalCountValues[i]);
        printf("%s", msg);
        OutputDebugString(msg);
    }
    DeleteCriticalSection(&CriticalSection);
#endif
}
