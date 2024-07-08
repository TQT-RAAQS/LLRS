#include "kernel.h"
#include <chrono>

double solve_gpu(int *sourceFlags, int *targetFlags, int numTraps, int numSources,
          int numTargets, int *OutSources_cpu, int *OutTargets_cpu) {
    // Allocate memory
    int *sourceFlags_d, *targetFlags_d, *OutSources_gpu_d, *OutTargets_gpu_d;
    int numExcessSources = numSources - numTargets;
    cudaMalloc((void **)&sourceFlags_d, numTraps * sizeof(int));
    cudaMalloc((void **)&targetFlags_d, numTraps * sizeof(int));
    cudaMalloc((void **)&OutSources_gpu_d, numTargets * sizeof(int));
    cudaMalloc((void **)&OutTargets_gpu_d, numTargets * sizeof(int));
    cudaDeviceSynchronize();
    // Copy data to GPU
    cudaMemcpy(sourceFlags_d, sourceFlags, numTraps * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(targetFlags_d, targetFlags, numTraps * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    // Compute on GPU
    auto t1 = std::chrono::high_resolution_clock::now();
    solve_gpu_d(numExcessSources, sourceFlags_d, targetFlags_d, numTraps,
                OutSources_gpu_d, OutTargets_gpu_d);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    double computeTime = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    // Copy data from GPU
    cudaMemcpy(OutSources_cpu, OutSources_gpu_d, numTargets * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(OutTargets_cpu, OutTargets_gpu_d, numTargets * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    // Free memory
    cudaFree(sourceFlags_d);
    cudaFree(targetFlags_d);
    cudaFree(OutSources_gpu_d);
    cudaFree(OutTargets_gpu_d);
    cudaDeviceSynchronize();
    return computeTime;
}