void lin_exact_cpu_v2_generate_matching(int* sourceFlags, int* targetFlags, int numTraps, int numSources, int numTargets, int* OutSources_cpu, int* OutTargets_cpu);
void lin_exact_1d_cpu_v2_block_output_generator(int matching_src[], int matching_dst[], int target_atom_count, int* src, int* dst, int* blk_sizes, int* batch_indices, int* num_batches, int* sol_length);
void lin_exact_cpu_v2_flags(int* sourceFlags, int* targetFlags, int numTraps, int numSources, int numTargets, int* OutSourcesFlags);
