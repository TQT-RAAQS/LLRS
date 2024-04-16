#include <cstdint>
void solve_gpu_d(int numExcessSources, int *sourceFlags, int *targetFlags,
                 int numTraps, int *OutSources_gpu_d, int *OutTargets_gpu_d);

typedef union {
    struct {
        int32_t key;
        int32_t value;
    };
    long long combined;
} KeyValue;
