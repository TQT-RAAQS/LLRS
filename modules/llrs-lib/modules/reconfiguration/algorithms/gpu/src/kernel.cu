#include "kernel.h"
#include <stdio.h>
#define BLOCK_DIM 1024
#define nextpow2(n)                                                            \
    (n <= 32)    ? 32                                                          \
    : (n <= 64)  ? 64                                                          \
    : (n <= 128) ? 128                                                         \
    : (n <= 256) ? 256                                                         \
    : (n <= 512) ? 512                                                         \
                 : 1024

__device__ void compute_heights_gpu(int *buffer_s, int numTraps,
                                    int *heights_s) {
    int *inBuffer_s = heights_s;
    int *outBuffer_s = buffer_s;
    for (unsigned int stride = 1; stride <= (nextpow2(numTraps)) / 2;
         stride *= 2) {
        outBuffer_s[threadIdx.x] =
            inBuffer_s[threadIdx.x] + ((threadIdx.x >= stride)
                                           ? (inBuffer_s[threadIdx.x - stride])
                                           : (0));
        __syncthreads();
        int *tmp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = tmp;
    }
    heights_s[threadIdx.x] = inBuffer_s[threadIdx.x];
}

__device__ void compressArray(int *buffer_s, int numTraps, int *flags_s,
                              int *Output_d) {

    int *inBuffer_s = flags_s;
    int *outBuffer_s = buffer_s;
    for (unsigned int stride = 1; stride <= (nextpow2(numTraps)) / 2;
         stride *= 2) {
        outBuffer_s[threadIdx.x] =
            inBuffer_s[threadIdx.x] + ((threadIdx.x >= stride)
                                           ? (inBuffer_s[threadIdx.x - stride])
                                           : (0));
        __syncthreads();
        int *tmp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = tmp;
    }

    if ((threadIdx.x == 0 && inBuffer_s[threadIdx.x] == 1) ||
        (threadIdx.x != 0 &&
         inBuffer_s[threadIdx.x] != inBuffer_s[threadIdx.x - 1])) {
        Output_d[inBuffer_s[threadIdx.x] - 1] = threadIdx.x;
    }
}

__device__ void linear_get_primes(int *sourceFlags_s, int *targetFlags_s,
                                  int numTraps, int *heights_s,
                                  int *y_prime_s) {

    y_prime_s[threadIdx.x] = numTraps - 1;
    int height_x = heights_s[threadIdx.x];
    int next_height_forward;
    int next_height_backward;
    int i_forward = threadIdx.x + 1;
    int i_backward = threadIdx.x - 1;
    int forward_flag = 0;
    int backward_flag = 0;

    if (targetFlags_s[threadIdx.x] == 1) {
        while (i_forward < numTraps || i_backward > -1) {

            if (forward_flag == 1 && backward_flag == 1) {
                break;
            }

            if (forward_flag == 1 && i_backward < 0) {
                break;
            }

            if (backward_flag == 1 && i_forward >= numTraps) {
                break;
            }

            if (forward_flag == 0 && i_forward < numTraps) {
                if (heights_s[i_forward] == height_x) {
                    if (sourceFlags_s[i_forward] == 1) {
                        y_prime_s[threadIdx.x] = i_forward;
                        forward_flag = 1;
                    } else {
                        i_forward++;
                    }
                } else {
                    next_height_forward = abs(heights_s[i_forward] - height_x);
                    i_forward += next_height_forward;
                }
            }

            if (backward_flag == 0 && i_backward > -1) {
                if (heights_s[i_backward] == height_x) {
                    if (sourceFlags_s[i_backward] == 1) {
                        y_prime_s[i_backward] = threadIdx.x;
                        backward_flag = 1;
                    } else {
                        i_backward--;
                    }
                } else {
                    i_backward--;
                    next_height_backward =
                        abs(heights_s[i_backward] - height_x);
                    i_backward -= next_height_backward;
                }
            }
        }
    }
}

__device__ void compute_profits_gpu(int *buffer_s, int *sourceFlags_s,
                                    int numExcessSources, int *heights_s,
                                    int numTraps, int *y_prime_s, int *profit_s,
                                    KeyValue *max_profit_s) {

    __shared__ int flag;

    int y_prime = y_prime_s[threadIdx.x];
    int next_source = y_prime_s[y_prime];
    int height_x = heights_s[threadIdx.x];
    int *inBuffer_s = profit_s;
    int *outBuffer_s = buffer_s;
    bool entered = 0;
    KeyValue val;

    if (sourceFlags_s[threadIdx.x] == 1) {
        if (height_x <= numExcessSources && height_x > 0) {
            inBuffer_s[threadIdx.x] = 2 * y_prime - next_source - threadIdx.x;
        }
    }

    if (threadIdx.x == 0) {
        flag = 1;
    }
    __syncthreads();
    int count = 1;

    while (flag == 1) {
        __syncthreads();
        flag = 0;
        if (sourceFlags_s[threadIdx.x] == 1) {
            if (heights_s[threadIdx.x] <= numExcessSources &&
                heights_s[threadIdx.x] > 0) {
                if (next_source < numTraps - 1) {
                    outBuffer_s[threadIdx.x] =
                        inBuffer_s[threadIdx.x] + inBuffer_s[next_source];
                    for (int i = 0; i < count; i++) {
                        next_source = y_prime_s[y_prime_s[next_source]];
                    }
                    count *= 2;
                    flag = 1;
                } else {

                    outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
                    if (entered == 0) {
                        val.key = BLOCK_DIM - threadIdx.x;
                        val.value = inBuffer_s[threadIdx.x];
                        atomicMax(&((&max_profit_s[height_x - 1])->combined),
                                  val.combined);
                        entered = 1;
                    }
                }
            }
        }
        __syncthreads();
        int *tmp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = tmp;
    }
    profit_s[threadIdx.x] = inBuffer_s[threadIdx.x];
}
__global__ void solve_gpu_kernel(int numExcessSources, int *sourceFlags,
                                 int *targetFlags, int numTraps,
                                 int *OutSources_gpu, int *OutTargets_gpu) {

    if (blockIdx.x == 0) {
        __shared__ int sourceFlags_s[BLOCK_DIM];
        __shared__ int targetFlags_s[BLOCK_DIM];
        __shared__ int outputFlags[BLOCK_DIM];
        __shared__ int heights_s[BLOCK_DIM];
        __shared__ int y_prime_s[BLOCK_DIM];
        __shared__ int profit_s[BLOCK_DIM];
        __shared__ KeyValue max_profit_s[BLOCK_DIM];

        int sourceFlag = sourceFlags[threadIdx.x];
        int targetFlag = targetFlags[threadIdx.x];
        outputFlags[threadIdx.x] = sourceFlag;

        if (sourceFlag == 1 && targetFlag == 1) {
            sourceFlags_s[threadIdx.x] = 0;
            targetFlags_s[threadIdx.x] = 0;
        } else {
            sourceFlags_s[threadIdx.x] = sourceFlag;
            targetFlags_s[threadIdx.x] = targetFlag;
        }
        profit_s[threadIdx.x] = 0;
        __syncthreads();

        heights_s[threadIdx.x] =
            sourceFlags_s[threadIdx.x] -
            ((threadIdx.x > 0) ? (targetFlags_s[threadIdx.x - 1]) : (0));
        if (threadIdx.x < numExcessSources) {
            KeyValue inf;
            inf.value = -INT_MAX;
            inf.key = numTraps - 1;
            max_profit_s[threadIdx.x].combined = inf.combined;
        }
        __syncthreads();

        __shared__ int buffer_s[BLOCK_DIM];
        compute_heights_gpu(buffer_s, numTraps, heights_s);
        __syncthreads();

        linear_get_primes(sourceFlags_s, targetFlags_s, numTraps, heights_s,
                          y_prime_s);
        __syncthreads();

        compute_profits_gpu(buffer_s, sourceFlags_s, numExcessSources,
                            heights_s, numTraps, y_prime_s, profit_s,
                            max_profit_s);
        __syncthreads();

        if (threadIdx.x < numExcessSources) {
            outputFlags[BLOCK_DIM - (&max_profit_s[threadIdx.x])->key] = 0;
        }
        __syncthreads();
        compressArray(buffer_s, numTraps, outputFlags, OutSources_gpu);
    }

    else {
        __shared__ int buffer_s[BLOCK_DIM];
        __shared__ int targetFlags_s[BLOCK_DIM];

        targetFlags_s[threadIdx.x] = targetFlags[threadIdx.x];
        __syncthreads();
        compressArray(buffer_s, numTraps, targetFlags_s, OutTargets_gpu);
    }
}

__global__ void populate_array_missing_sources(int *OutSources_gpu,
                                               int *targetFlags, int numTraps,
                                               int *OutTargets_gpu) {

    __shared__ int buffer_s[BLOCK_DIM];
    __shared__ int targetFlags_s[BLOCK_DIM];

    targetFlags_s[threadIdx.x] = targetFlags[threadIdx.x];

    __syncthreads();

    int *inBuffer_s = targetFlags_s;
    int *outBuffer_s = buffer_s;
    for (unsigned int stride = 1; stride <= (nextpow2(numTraps)) / 2;
         stride *= 2) {
        outBuffer_s[threadIdx.x] =
            inBuffer_s[threadIdx.x] + ((threadIdx.x >= stride)
                                           ? (inBuffer_s[threadIdx.x - stride])
                                           : (0));
        __syncthreads();
        int *tmp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = tmp;
    }

    if ((threadIdx.x == 0 && inBuffer_s[threadIdx.x] == 1) ||
        (threadIdx.x != 0 &&
         inBuffer_s[threadIdx.x] != inBuffer_s[threadIdx.x - 1])) {
        OutTargets_gpu[inBuffer_s[threadIdx.x] - 1] = threadIdx.x;
        OutSources_gpu[inBuffer_s[threadIdx.x] - 1] = -1;
    }
}

__global__ void populate_array_done(int numTraps, int *sourceFlags,
                                    int *OutSources_gpu, int *targetFlags,
                                    int *OutTargets_gpu) {

    if (blockIdx.x == 0) {
        __shared__ int buffer_s[BLOCK_DIM];
        __shared__ int sourceFlags_s[BLOCK_DIM];

        sourceFlags_s[threadIdx.x] = sourceFlags[threadIdx.x];
        __syncthreads();

        compressArray(buffer_s, numTraps, sourceFlags_s, OutSources_gpu);
    }

    else {
        __shared__ int buffer_s[BLOCK_DIM];
        __shared__ int targetFlags_s[BLOCK_DIM];

        targetFlags_s[threadIdx.x] = targetFlags[threadIdx.x];
        __syncthreads();

        compressArray(buffer_s, numTraps, targetFlags_s, OutTargets_gpu);
    }
}

__host__ void solve_gpu_d(int numExcessSources, int *sourceFlags,
                          int *targetFlags, int numTraps, int *OutSources_gpu,
                          int *OutTargets_gpu) {

    if (numExcessSources > 0) {
        solve_gpu_kernel<<<2, numTraps>>>(numExcessSources, sourceFlags,
                                          targetFlags, numTraps, OutSources_gpu,
                                          OutTargets_gpu);
    } else if (numExcessSources == 0) {
        populate_array_done<<<2, numTraps>>>(
            numTraps, sourceFlags, OutSources_gpu, targetFlags, OutTargets_gpu);
    } else {
        populate_array_missing_sources<<<1, numTraps>>>(
            OutSources_gpu, targetFlags, numTraps, OutTargets_gpu);
    }
}