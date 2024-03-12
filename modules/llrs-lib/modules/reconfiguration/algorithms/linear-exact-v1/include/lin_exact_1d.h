#ifndef LINEAR_EXACT_H
#define LINEAR_EXACT_H
void lin_exact_generate_matching(int* initial, int* target, int num_traps, int initial_atom_count, int target_atom_count, int matching_src[], int matching_dst[]);
void block_output_generator_cpu(int matching_src[], int matching_dst[], int target_atom_count, int* src, int* dst, int* blk_sizes, int* batch_indices, int* num_batches, int* sol_length);

#endif