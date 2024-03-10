#include <ctime>
#include <chrono>
#include "gpufun.h"
#include "kernel.h"
#include "timer.h"

//#define VERBOSE 

//#define PRINT_GPU_RUNTIME


std::chrono::duration<double, std::milli> solve_gpu(int* sourceFlags, int* targetFlags, int numTraps, int numSources, int numTargets, int* OutSources_gpu, int* OutTargets_gpu)
{
    Timer timer;
	//float computeTimer;

    std::chrono::time_point<std::chrono::high_resolution_clock> t1;
    std::chrono::time_point<std::chrono::high_resolution_clock> t2;

    std::chrono::duration<double, std::milli> computeTimer;
	
    // Allocate memory
    startTime(&timer);
    int *sourceFlags_d, *targetFlags_d, *OutSources_gpu_d, *OutTargets_gpu_d;
    int numExcessSources = numSources - numTargets;
    cudaMalloc((void**) &sourceFlags_d, numTraps*sizeof(int));
    cudaMalloc((void**) &targetFlags_d, numTraps*sizeof(int));
    cudaMalloc((void**) &OutSources_gpu_d, numTargets*sizeof(int));
    cudaMalloc((void**) &OutTargets_gpu_d, numTargets*sizeof(int));
    cudaDeviceSynchronize();
    stopTime(&timer);
#ifdef VERBOSE
    printElapsedTime(timer, "\nMy Allocation time", DGREEN);
#endif
    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(sourceFlags_d, sourceFlags, numTraps*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(targetFlags_d, targetFlags, numTraps*sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);

#ifdef VERBOSE
    printElapsedTime(timer, "My Copy to GPU time", DGREEN);
#endif


    // Compute on GPU
    t1 = std::chrono::high_resolution_clock::now();
    solve_gpu_d(numExcessSources, sourceFlags_d, targetFlags_d, numTraps, OutSources_gpu_d, OutTargets_gpu_d);
    cudaDeviceSynchronize();
	t2 = std::chrono::high_resolution_clock::now();
    computeTimer = t2 - t1;
#ifdef PRINT_GPU_RUNTIME
    printElapsedTime(timer, "GPU time is:", CYAN);
#endif

    // Copy data from GPU
    startTime(&timer);
    cudaMemcpy(OutSources_gpu, OutSources_gpu_d, numTargets*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(OutTargets_gpu, OutTargets_gpu_d, numTargets*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
#ifdef VERBOSE
    printElapsedTime(timer, "My Copy from GPU time", DGREEN);
#endif
    // Free memory
    startTime(&timer);
    cudaFree(sourceFlags_d);
    cudaFree(targetFlags_d);
    cudaFree(OutSources_gpu_d);
	cudaFree(OutTargets_gpu_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
#ifdef VERBOSE
    printElapsedTime(timer, "Deallocation time", DGREEN);
#endif
	
	return computeTimer;

}