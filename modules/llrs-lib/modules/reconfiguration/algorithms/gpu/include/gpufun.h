std::chrono::duration<double, std::milli>
solve_gpu(int *sourceFlags, int *targetFlags, int numTraps, int numSources,
          int numTargets, int *OutSources_gpu, int *OutTargets_gpu);