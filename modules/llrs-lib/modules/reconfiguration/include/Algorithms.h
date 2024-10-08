#ifndef ALGORITHMS_H
#define ALGORITHMS_H
/*
The declarations and repackagings for the algorithms that are used in the Solver
module
*/

extern "C" void lin_exact_block_batched_cpu(
    int *initial, int *target, int num_traps, int inital_atom_count,
    int target_atom_count, int *src, int *dst, int *blk_sizes,
    int *batch_indices, int *num_batches, int *sol_length);

void solve_cpu(int *sourceFlags, int *targetFlags, int numTraps, int numSources,
               int numTargets, int *OutSources_cpu, int *OutTargets_cpu,
               int *outputMovesSource, int *outputMovesTarget,
               int *movesCounter);

extern "C" void lin_exact_1d_cpu_v2_block_output_generator(
    int matching_src[], int matching_dst[], int target_atom_count, int *src,
    int *dst, int *blk_sizes, int *batch_indices, int *num_batches,
    int *sol_length);

float solve_gpu(int *sourceFlags, int *targetFlags, int numTraps,
                int numSources, int numTargets, int *outSources,
                int *outTargets, int *outputMovesSource, int *outputMovesTarget,
                int *movesCounter);

inline void redrec_v2(int *initial, int initial_atom_count, int Nt_x, int Nt_y,
                      int R_h, int *src, int *dst, int *batch_indices,
                      int *num_batches, int *sol_length) {}

extern "C" void redrec_cpu(int gridHeight, int width, int reservoirHeight,
                           int *sourceFlags, int *OutSources_cpu,
                           int *OutTargets_cpu, int *outputMovesSource_cpu,
                           int *outputMovesTarget_cpu, int *moves_counter,
                           int *path_system, int *path_length);

extern "C" double redrec_gpu(int gridHeight, int width, int reservoirHeight,
                             int *sourceFlags, int *outSources, int *outTargets,
                             int *outputMovesSource, int *outputMovesTarget,
                             int *movesCounter, int *pathSystem,
                             int *pathSystemLength);

extern "C" void bird_cpu(int gridHeight, int width, int reservoirHeight,
                         int *sourceFlags, int *outSources, int *outTargets,
                         int *outputMovesSource, int *outputMovesTarget,
                         int *movesCounter, int *pathSystem,
                         int *pathSystemLength);

extern "C" void aro_serial_cpu(int *source, int *target, int H, int W, int K,
                               int K_prime, int rerouting, int atom_isolation,
                               int ordering, int *src, int *dst,
                               int *sol_length, int *path_system,
                               int *path_length);

extern "C" void hungarian_colav(
    int *source, int *target, int H, int W, int K, int K_prime, int use_weights,
    int no_token_w, int one_token_w, int two_token_w,
    int collision_permissiveness, int margin, int collision_avoidance_type,
    int order_moves, int use_greedy_isolation, int use_unit_edge_weights,
    int output_hungarian_colav_parameters, int *src_batched, int *dst_batched,
    int *batchPtr_batched, int *num_batches_batched, int *sol_length_batched);
#endif