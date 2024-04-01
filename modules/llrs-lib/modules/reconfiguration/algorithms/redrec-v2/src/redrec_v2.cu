// Author: Remy El Sabeh
#define INF ~(1 << 31) / 2

#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

Timer timer;
float batching_time = 0;

int abs(int a) {
    if (a < 0)
        return -a;
    return a;
}

void random_shuffle(int *a, int N) {
    for (unsigned int i = N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }
}

void populate_array_of_atoms(int *a, int num_atoms, int N) {
    for (unsigned int i = 0; i < N; ++i) {
        if (i < num_atoms) {
            a[i] = 1;
        } else {
            a[i] = 0;
        }
    }
    random_shuffle(a, N);
}

void populate_hungarian_indices_to_original_indices(
    int *source, int *target, int N, int hungarian_index_to_source_index[],
    int hungarian_index_to_target_index[]) {
    int hungarian_source_index = 0;
    int hungarian_target_index = 0;

    for (int i = 0; i < N; i++) {
        if (source[i] == 1) {
            hungarian_index_to_source_index[hungarian_source_index++] = i;
        }
        if (target[i] == 1) {
            hungarian_index_to_target_index[hungarian_target_index++] = i;
        }
    }
}

int is_vertical(int u, int v, int W) {
    int u_col = u % W;
    int v_col = v % W;
    return u_col == v_col;
}

int verify_solution(int *source, int *target, int len) {
    for (int i = 0; i < len; i++) {
        if (target[i] == 1) {
            if (source[i] != 1)
                return 0;
        }
    }
    return 1;
}

void populate_edge_weights_with_unit_weight(int *edge_weights, int num_edges) {
    for (int i = 0; i < num_edges; i++) {
        edge_weights[i] = 1;
    }
}

void copy_array(int *arr1, int *arr2, int length) {
    for (int i = 0; i < length; i++) {
        arr2[i] = arr1[i];
    }
}

void print_grid(int *grid, int H, int W) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            printf("%d ", grid[i * W + j]);
        }
        printf("\n");
    }
}

void batched_output_generator_cpu(int matching_src[], int matching_dst[],
                                  int K_prime, int *src_batched,
                                  int *dst_batched, int *batchPtr_batched,
                                  int *sol_length_batched,
                                  int *num_batches_batched) {
    startTime(&timer);
    // For each element of the matching, compute the distance from source to
    // target
    int distance_to_target[K_prime];

    for (int i = 0; i < K_prime; i++) {
        distance_to_target[i] = abs(matching_src[i] - matching_dst[i]);
    }

    // Get the maximum distance from any source to any target
    int max_distance = 0;
    for (int i = 0; i < K_prime; i++) {
        max_distance = max(max_distance, distance_to_target[i]);
    }

    *num_batches_batched = max_distance;

    int total_moves = 0;

    // max_distance defines the number of batches that we have
    for (int batch_index = 0; batch_index < max_distance; batch_index++) {
        batchPtr_batched[batch_index] = total_moves;
        for (int i = 0; i < K_prime; i++) {
            if (batch_index < distance_to_target[i]) {
                if (matching_src[i] < matching_dst[i]) {
                    src_batched[total_moves] = matching_src[i];
                    dst_batched[total_moves] = matching_src[i] + 1;
                    matching_src[i]++;
                    total_moves++;
                } else {
                    src_batched[total_moves] = matching_src[i];
                    dst_batched[total_moves] = matching_src[i] - 1;
                    matching_src[i]--;
                    total_moves++;
                }
            }
        }
    }

    *sol_length_batched = total_moves;
    stopTime(&timer);
    batching_time += getElapsedTime(timer);
}

// Given a matching, calculate total displacement
int get_cost(int compressed_src[], int starting_index, int target_start,
             int K_prime) {
    int total_cost = 0;
    for (int i = 0; i < K_prime; i++) {
        total_cost +=
            abs(compressed_src[starting_index + i] - (target_start + i));
    }
    return total_cost;
}

// Generates optimal matching and returns its cost
// Corner cases:
// 1- Everything is before start of target: handled by
// first_atom_after_target_start = K; 2- Everything is after start of target:
// startIndex will be equal to endIndex Note that startIndex >= endIndex
void bruteforce_generate_matching(int *source, int N, int K, int K_prime,
                                  int K_1, int K_2, int matching_src[],
                                  int matching_dst[]) {
    // Compress source array
    int compressed_src[K];
    int compressed_idx = 0;
    // This handles the case where there are no atoms after target start (when
    // this is set to K, startingIndex = endingIndex = K - K_prime)
    int first_atom_after_target_start = K;
    int atoms_before_target = 0;
    int atoms_in_target = 0;
    // Target boundaries (inclusive)
    int target_start = K_1;
    int target_end = N - K_2 - 1;
    for (int i = 0; i < N; i++) {
        if (source[i] == 1) {
            compressed_src[compressed_idx] = i;
            if (i < target_start) {
                atoms_before_target++;
            } else if (i > target_end) {
                if (first_atom_after_target_start == K) {
                    first_atom_after_target_start = compressed_idx;
                }
            } else {
                atoms_in_target++;
                if (first_atom_after_target_start == K) {
                    first_atom_after_target_start = compressed_idx;
                }
            }
            compressed_idx++;
        }
    }

    int deficit = K_prime - atoms_in_target;

    int min_cost = INF;
    int selected_index = -1;

    // Ensure that there are at least K_prime atoms to the right, so if
    // first_atom_after_target_start is too "to the right", the first split
    // might not be 0 atoms from before target, it might be more
    int startingIndex = min(K - K_prime, first_atom_after_target_start);
    // This part is clear: starting from leftmost atom after target start,
    // select up to decifit/atoms_before_target atoms from before target start,
    // whichever is smaller
    int endingIndex =
        first_atom_after_target_start - min(atoms_before_target, deficit);

    for (int i = startingIndex; i >= endingIndex; i--) {
        int cost = get_cost(compressed_src, i, target_start, K_prime);
        if (cost < min_cost) {
            min_cost = cost;
            selected_index = i;
        }
    }

    int cur_target = K_1;
    for (int i = 0; i < K_prime; i++) {
        matching_src[i] = compressed_src[selected_index++];
        matching_dst[i] = cur_target++;
    }
}

void bruteforce_generate_moves_batched_cpu(int *source, int N, int K,
                                           int K_prime, int K_1, int K_2,
                                           int *src_batched, int *dst_batched,
                                           int *batchPtr_batched,
                                           int *num_batches_batched,
                                           int *sol_length_batched) {
    // Matching that corresponds to the best split
    int matching_src[K_prime];
    int matching_dst[K_prime];

    bruteforce_generate_matching(source, N, K, K_prime, K_1, K_2, matching_src,
                                 matching_dst);

    batched_output_generator_cpu(matching_src, matching_dst, K_prime,
                                 src_batched, dst_batched, batchPtr_batched,
                                 sol_length_batched, num_batches_batched);
}

// This function takes in results from a 1D solver call, converts them, executes
// them, then adds them to the overall output of the 2D solver
void convert_and_execute_1d_solver_output(
    int *source, int W, int d_c, int *src_batched_1d, int *dst_batched_1d,
    int *batchPtr_batched_1d, int num_batches_batched_1d,
    int sol_length_batched_1d, int *src_batched, int *dst_batched,
    int *batchPtr_batched, int *num_batches_batched, int *sol_length_batched) {
    startTime(&timer);
    int local_displacements_executed = 0;

    // Make sure to reindex whenever necessary
    for (int i = 0; i < num_batches_batched_1d; i++) {
        // Save batch ptr
        batchPtr_batched[*num_batches_batched + i] =
            *sol_length_batched + local_displacements_executed;
        for (int j = batchPtr_batched_1d[i];
             j < (i == num_batches_batched_1d - 1 ? sol_length_batched_1d
                                                  : batchPtr_batched_1d[i + 1]);
             j++) {
            int src = src_batched_1d[j];
            int dst = dst_batched_1d[j];
            // Update the configuration
            source[src * W + d_c]--;
            source[dst * W + d_c]++;
            // Add the move to the output
            src_batched[*sol_length_batched + local_displacements_executed] =
                src * W + d_c;
            dst_batched[*sol_length_batched + local_displacements_executed] =
                dst * W + d_c;
            // Increment the number of local displacements
            local_displacements_executed++;
        }
    }

    // Update the overall number of batches and length of the solution
    *sol_length_batched += local_displacements_executed;
    *num_batches_batched += num_batches_batched_1d;
    stopTime(&timer);
    batching_time += getElapsedTime(timer);
}

// This function is almost the same as the one above, except it does not do any
// conversion This function is used on the output of shuffling
void execute_shuffling_output(int *source, int *src_batched_1d,
                              int *dst_batched_1d, int *batchPtr_batched_1d,
                              int num_batches_batched_1d,
                              int sol_length_batched_1d, int *src_batched,
                              int *dst_batched, int *batchPtr_batched,
                              int *num_batches_batched,
                              int *sol_length_batched) {
    startTime(&timer);
    int local_displacements_executed = 0;

    for (int i = 0; i < num_batches_batched_1d; i++) {
        batchPtr_batched[*num_batches_batched + i] =
            *sol_length_batched + local_displacements_executed;
        for (int j = batchPtr_batched_1d[i];
             j < (i == num_batches_batched_1d - 1 ? sol_length_batched_1d
                                                  : batchPtr_batched_1d[i + 1]);
             j++) {
            int src = src_batched_1d[j];
            int dst = dst_batched_1d[j];
            // Update the configuration
            source[src]--;
            source[dst]++;
            // Add the move to the output
            src_batched[*sol_length_batched + local_displacements_executed] =
                src;
            dst_batched[*sol_length_batched + local_displacements_executed] =
                dst;
            // Increment the number of local displacements
            local_displacements_executed++;
        }
    }

    // Update the overall number of batches and length of the solution
    *sol_length_batched += local_displacements_executed;
    *num_batches_batched += num_batches_batched_1d;
    stopTime(&timer);
    batching_time += getElapsedTime(timer);
}

int get_excess(int *source, int *target, int N) {
    int e = 0;
    for (int i = 1; i <= N; i++) {
        if (source[i])
            e++;
        if (target[i])
            e--;
    }
    return e;
}

void compute_heights(int *source, int *target, int N, int H[], int *e) {

    int cur_num_sources_inclusive = 0;
    int cur_num_target_exclusive = 0;

    H[0] = 0;

    for (int i = 1; i <= N; i++) {
        cur_num_sources_inclusive += source[i];
        cur_num_target_exclusive += target[i - 1];
        H[i] = cur_num_sources_inclusive - cur_num_target_exclusive;
    }

    H[N + 1] = (*e) = get_excess(source, target, N);
}

// matching generator
void lin_exact_generate_matching(int *source, int *target, int N, int K,
                                 int K_prime, int matching_src[],
                                 int matching_dst[]) {
    int X = N + 1;
    // Including 0; H[0] = 0 and H[N + 1] = e
    int H[N + 2];

    // Offset the source just so that the sources and targets are in (0, X) (set
    // X to N + 1)
    int source_with_offset[N + 1];
    int target_with_offset[N + 1];

    source_with_offset[0] = 0;
    target_with_offset[0] = 0;
    for (int i = 1; i <= N; i++) {
        source_with_offset[i] = source[i - 1];
        target_with_offset[i] = target[i - 1];
    }

    // Compress source and target
    int S_set[K];
    int S_set_idx;
    S_set_idx = 0;

    for (int i = 1; i <= N; i++) {
        if (source_with_offset[i]) {
            S_set[S_set_idx++] = i;
        }
    }

    int e;

    compute_heights(source_with_offset, target_with_offset, N, H, &e);

    int h_max, h_min;

    h_max = h_min = H[0];

    for (int i = 0; i <= X; i++) {
        h_max = max(h_max, H[i]);
        h_min = min(h_min, H[i]);
    }

    // Those are only defined for [h_min...h_max] so make sure to index properly
    // The way this will work is that S[k] will be an array of indices
    int S_tilde[h_max - h_min + 1][N];
    int T_tilde[h_max - h_min + 1][N];
    int S_tilde_indices[h_max - h_min + 1];
    int T_tilde_indices[h_max - h_min + 1];
    // We can get the corresponding k using H, the below arrays store the
    // indices within the k
    int source_to_S_tilde_index[X];

    for (int i = 0; i < h_max - h_min + 1; i++) {
        S_tilde_indices[i] = 0;
        T_tilde_indices[i] = 0;
    }

    // We do this to preserve the order within every k for S_tilde
    for (int k = h_min; k <= h_max; k++) {
        if (k <= 0) {
            S_tilde[k - h_min][S_tilde_indices[k - h_min]++] = 0;
        }
    }

    for (int i = 1; i <= N; i++) {
        if (source_with_offset[i]) {
            S_tilde[H[i] - h_min][S_tilde_indices[H[i] - h_min]] = i;
            // Map i to S_tilde index
            source_to_S_tilde_index[i] = S_tilde_indices[H[i] - h_min];
            // Increment indices
            S_tilde_indices[H[i] - h_min]++;
        }
        if (target_with_offset[i]) {
            T_tilde[H[i] - h_min][T_tilde_indices[H[i] - h_min]] = i;
            // Increment indices
            T_tilde_indices[H[i] - h_min]++;
        }
    }

    // Add the X where necessary
    for (int k = h_min; k <= h_max; k++) {
        if (k <= e) {
            T_tilde[k - h_min][T_tilde_indices[k - h_min]++] = X;
        }
        // This is for the corner case that we had to take care of, may or may
        // not work
        else {
            S_tilde[k - h_min][S_tilde_indices[k - h_min]++] = X;
        }
    }

    int P[X + 1];

    P[X] = 0;

    // Loop over the elements in S
    for (int i = K - 1; i >= 0; i--) {
        // Get y
        int y = S_set[i];
        // Get height of y
        int height = H[y];
        // Get adjusted height
        int k = height - h_min;
        // We are looking for y', which is defined for a y in S_tilde as
        // whatever is in T_tilde at the same index
        int y_prime;
        if (y != X) {
            y_prime = T_tilde[k][source_to_S_tilde_index[y]];
        } else {
            // Corner case: X' = X (that will never happen here because S_set[i]
            // will never be equal to X, but it may happen for X'')
            y_prime = X;
        }
        // We are also looking for y'', which is defined for a y' in T_tilde as
        // whatever is in S_tilde at the next index
        int y_double_prime;
        if (y_prime != X) {
            y_double_prime = S_tilde[k][source_to_S_tilde_index[y] + 1];
        } else {
            // Corner case: X' = X
            y_double_prime = X;
        }
        // Finally, compute P
        P[y] = 2 * y_prime - y - y_double_prime + P[y_double_prime];
    }

    // pi is not defined for h_min but for ease of indexing we will assume it is
    int pi[h_max - h_min + 1];
    // Initialize to -INF
    for (int i = 0; i < h_max - h_min + 1; i++) {
        pi[i] = -INF;
    }

    // Go over all the Ps and find the max of every pi[k]
    for (int i = 0; i < K; i++) {
        int y = S_set[i];
        int k = H[y];
        pi[k - h_min] = max(pi[k - h_min], P[y]);
    }

    int to_be_removed[N + 1];
    int max_elem[h_max - h_min + 1];
    int max_value[h_max - h_min + 1];

    for (int i = 1; i <= N; i++) {
        to_be_removed[i] = 0;
    }

    for (int i = 0; i < h_max - h_min + 1; i++) {
        max_elem[i] = -1;
        max_value[i] = -INF;
    }

    // pi contains the maximum profit needed for heights between 1 and e
    // (inclusive), so we just loop over the values now!
    for (int i = 0; i < K; i++) {
        int y = S_set[i];
        if (P[y] > max_value[H[y] - h_min]) {
            max_elem[H[y] - h_min] = y;
            max_value[H[y] - h_min] = P[y];
        }
    }

    // Populate to_be_removed
    for (int i = 1; i <= e; i++) {
        to_be_removed[max_elem[i - h_min]] = 1;
    }

    int matching_src_index = 0;
    int matching_dst_index = 0;

    // Write to output, do not forget to offset back to make things 0-indexed
    // again
    for (int i = 1; i <= N; i++) {
        if (source_with_offset[i] && !to_be_removed[i]) {
            matching_src[matching_src_index++] = i - 1;
        }
        if (target_with_offset[i]) {
            matching_dst[matching_dst_index++] = i - 1;
        }
    }
}

void lin_exact_generate_moves_batched_cpu(int *source, int *target, int N,
                                          int K, int K_prime, int *src_batched,
                                          int *dst_batched,
                                          int *batchPtr_batched,
                                          int *num_batches_batched,
                                          int *sol_length_batched) {

    int matching_src[K_prime];
    int matching_dst[K_prime];

    lin_exact_generate_matching(source, target, N, K, K_prime, matching_src,
                                matching_dst);

    batched_output_generator_cpu(matching_src, matching_dst, K_prime,
                                 src_batched, dst_batched, batchPtr_batched,
                                 sol_length_batched, num_batches_batched);
}

int get_top_reservoir_displacement(int *source, int *atoms_selected_from_top,
                                   int num_traps_selected_from_top, int R_h,
                                   int H, int W, int r_c) {
    int displacement = 0;
    int cur_row = R_h;
    int cur_trap = 0;
    int traps_left = num_traps_selected_from_top;
    while (traps_left > 0) {
        // Found a vacant spot
        if (source[cur_row * W + r_c] == 0) {
            // Match the current atom to the vacant spot
            displacement += abs(atoms_selected_from_top[cur_trap] - cur_row);
            cur_trap++;
            traps_left--;
        }
        cur_row++;
    }
    return displacement;
}

int get_bottom_reservoir_displacement(int *source,
                                      int *atoms_selected_from_bottom,
                                      int num_traps_selected_from_bottom,
                                      int R_h, int H, int W, int r_c) {
    int displacement = 0;
    int cur_row = H - R_h - 1;
    int cur_trap = num_traps_selected_from_bottom - 1;
    int traps_left = num_traps_selected_from_bottom;
    while (traps_left > 0) {
        // Found a vacant spot
        if (source[cur_row * W + r_c] == 0) {
            // Match the current atom to the vacant spot
            displacement += abs(atoms_selected_from_bottom[cur_trap] - cur_row);
            cur_trap--;
            traps_left--;
        }
        cur_row--;
    }
    return displacement;
}

// Given a row and a column, return the index of the trap
int coord_to_idx(int r, int c, int W) { return r * W + c; }

// Given a trap index, get its row and column
void idx_to_coord(int idx, int W, int *r, int *c) {
    *r = idx / W;
    *c = idx % W;
}

void update_dynamic_columns(int W, int *column_solved, int *dynamic_columns) {
    int leftmost_unsolved_column = -1;
    for (int c = W - 1; c >= 0; c--) {
        if (!column_solved[c]) {
            leftmost_unsolved_column = c;
        }
        dynamic_columns[c] = leftmost_unsolved_column;
    }
}

void compute_surplus(int *source, int H, int W, int T_h, int *surplus) {
    for (int c = 0; c < W; c++) {
        int atoms_in_col = 0;
        for (int r = 0; r < H; r++) {
            if (source[r * W + c] == 1) {
                atoms_in_col += 1;
            }
        }
        surplus[c] = atoms_in_col - T_h;
    }
}

int get_vacant_reservoir_traps(int *source, int H, int W, int R_h, int c) {
    int num_vacant_traps = 0;
    for (int r = 0; r < R_h; r++) {
        if (source[r * W + c] == 0) {
            num_vacant_traps += 1;
        }
    }

    for (int r = H - R_h; r < H; r++) {
        if (source[r * W + c] == 0) {
            num_vacant_traps += 1;
        }
    }

    return num_vacant_traps;
}

// This is very similar to the batched output generator of the 1D Solvers
void generate_and_execute_center_batches(
    int *source, int *matching_src, int *matching_dst, int W,
    int matching_length, int *src_batched, int *dst_batched,
    int *batchPtr_batched, int *num_batches_batched, int *sol_length_batched) {
    startTime(&timer);
    // Get the required number of batches
    int local_num_batches = 0;
    for (int i = 0; i < matching_length; i++) {
        local_num_batches =
            max(local_num_batches, abs(matching_src[i] - matching_dst[i]) / W);
    }

    // Keep track of the total number of displacements in this function call
    int local_displacements = 0;

    for (int batch_index = 0; batch_index < local_num_batches; batch_index++) {
        batchPtr_batched[*num_batches_batched + batch_index] =
            *sol_length_batched + local_displacements;
        for (int i = 0; i < matching_length; i++) {
            if (matching_src[i] < matching_dst[i]) {
                src_batched[*sol_length_batched + local_displacements] =
                    matching_src[i];
                dst_batched[*sol_length_batched + local_displacements] =
                    matching_src[i] + W;

                // Execute displacement
                source[matching_src[i]]--;
                source[matching_src[i] + W]++;

                matching_src[i] += W;
                local_displacements++;
            } else if (matching_src[i] > matching_dst[i]) {
                src_batched[*sol_length_batched + local_displacements] =
                    matching_src[i];
                dst_batched[*sol_length_batched + local_displacements] =
                    matching_src[i] - W;

                // Execute displacement
                source[matching_src[i]]--;
                source[matching_src[i] - W]++;

                matching_src[i] -= W;
                local_displacements++;
            }
        }
    }

    // Update the overall solution length
    *sol_length_batched += local_displacements;
    // Update the overall number of batches
    *num_batches_batched += local_num_batches;
    stopTime(&timer);
    batching_time += getElapsedTime(timer);
}

void generate_centered_matching(int *source, int H, int W, int c,
                                int starting_row, int *matching_src,
                                int *matching_dst) {
    int cur_row = starting_row;
    int matching_idx = 0;
    for (int r = 0; r < H; r++) {
        if (source[r * W + c] == 1) {
            matching_src[matching_idx] = coord_to_idx(r, c, W);
            matching_dst[matching_idx] = coord_to_idx(cur_row, c, W);
            matching_idx++;
            cur_row++;
        }
    }
}

void center_sort(int *source, int c, int num_atoms_in_c, int starting_row,
                 int H, int W, int *src_batched, int *dst_batched,
                 int *batchPtr_batched, int *num_batches_batched,
                 int *sol_length_batched) {
    // This is where we will be storing the matching
    int matching_src[num_atoms_in_c];
    int matching_dst[num_atoms_in_c];

    generate_centered_matching(source, H, W, c, starting_row, matching_src,
                               matching_dst);

    // Generate batches, add them to output and execute them
    generate_and_execute_center_batches(
        source, matching_src, matching_dst, W, num_atoms_in_c, src_batched,
        dst_batched, batchPtr_batched, num_batches_batched, sol_length_batched);
}

void partial_sort(int *source, int c, int *surplus, int T_h, int H, int W,
                  int *src_batched, int *dst_batched, int *batchPtr_batched,
                  int *num_batches_batched, int *sol_length_batched) {
    // Get the number of atoms in the column
    int num_atoms_in_c = surplus[c] + T_h;
    // Now that we know how many atoms we have in the column, we know
    // where the center region begins
    center_sort(source, c, num_atoms_in_c, (H - num_atoms_in_c) / 2, H, W,
                src_batched, dst_batched, batchPtr_batched, num_batches_batched,
                sol_length_batched);
}

void sort_net_zero_column(int *source, int c, int T_h, int R_h, int H, int W,
                          int *dynamic_columns, int *column_solved,
                          int *src_batched, int *dst_batched,
                          int *batchPtr_batched, int *num_batches_batched,
                          int *sol_length_batched) {
    // If we are here, we know exactly how many atoms there are in the column
    // (T_h) and we know where center sort should start from (R_h)
    center_sort(source, c, T_h, R_h, H, W, src_batched, dst_batched,
                batchPtr_batched, num_batches_batched, sol_length_batched);
    // Set current column to be solved
    column_solved[c] = 1;
    // Update dynamic columns
    update_dynamic_columns(W, column_solved, dynamic_columns);
}

void sort_net_zero_columns(int *source, int H, int W, int T_h, int R_h,
                           int *surplus, int *dynamic_columns,
                           int *column_solved, int *src_batched,
                           int *dst_batched, int *batchPtr_batched,
                           int *num_batches_batched, int *sol_length_batched) {
    for (int c = 0; c < W; c++) {
        if (surplus[c] == 0) {
            sort_net_zero_column(source, c, T_h, R_h, H, W, dynamic_columns,
                                 column_solved, src_batched, dst_batched,
                                 batchPtr_batched, num_batches_batched,
                                 sol_length_batched);
        }
    }
}

int deficit_exists(int *surplus, int W) {
    for (int c = 0; c < W; c++) {
        if (surplus[c] < 0) {
            return 1;
        }
    }
    return 0;
}

void find_pair_to_shuffle(int *surplus, int *column_solved,
                          int *dynamic_columns, int *d, int *r, int W) {
    int atoms_to_shuffle = -INF;
    for (int i = 0; i < W - 1; i++) {
        // Check if the current column has not been solved and that it has an
        // unsolved column to its right, then check whether they make up a valid
        // donor-receiver pair
        if (!column_solved[i] && dynamic_columns[i + 1] != -1 &&
            (surplus[i] * surplus[dynamic_columns[i + 1]] < 0)) {
            atoms_to_shuffle =
                max(atoms_to_shuffle,
                    min(abs(surplus[i]), abs(surplus[dynamic_columns[i + 1]])));
        }
    }
    // Now that we know the maximum number of atoms that we can shuffle, we pick
    // the pair that minimizes the difference of the absolute values, i.e. that
    // gets both columns as close to completion as possible
    int selected_pair_index = -1;
    int abs_val_diff = INF;
    for (int i = 0; i < W - 1; i++) {
        if (!column_solved[i] && dynamic_columns[i + 1] != -1 &&
            (surplus[i] * surplus[dynamic_columns[i + 1]] < 0) &&
            (min(abs(surplus[i]), abs(surplus[dynamic_columns[i + 1]])) ==
             atoms_to_shuffle)) {
            // We have a valid candidate
            int max_abs_val =
                max(abs(surplus[i]), abs(surplus[dynamic_columns[i + 1]]));
            int min_abs_val =
                min(abs(surplus[i]), abs(surplus[dynamic_columns[i + 1]]));
            if (max_abs_val - min_abs_val < abs_val_diff) {
                selected_pair_index = i;
                abs_val_diff = max_abs_val - min_abs_val;
            }
        }
    }
    if (surplus[selected_pair_index] > 0) {
        *d = selected_pair_index;
        *r = dynamic_columns[selected_pair_index + 1];
    } else {
        *d = dynamic_columns[selected_pair_index + 1];
        *r = selected_pair_index;
    }
}

// This is the variant of the function above that we use for REDREC v2.1
void find_pairs_to_shuffle(int *surplus, int *column_solved,
                           int *dynamic_columns, int *donors, int *receivers,
                           int *num_pairs_to_shuffle,
                           int max_num_pairs_to_shuffle, int W) {
    *num_pairs_to_shuffle = 0;

    // This is a mapping from the compressed array to the original indices
    int compressed_to_original[W];
    int compressed_to_original_idx = 0;

    // Here, we will save the atoms all pairs can shuffle (then we will use this
    // information to sort the pairs)
    int atoms_to_shuffle[W];
    // We will also save the absolute value differences in case a tiebreaker is
    // needed
    int abs_val_diff[W];

    for (int i = 0; i < W - 1; i++) {
        // Check if the current column has not been solved and that it has an
        // unsolved column to its right, then check whether they make up a valid
        // donor-receiver pair
        if (!column_solved[i] && dynamic_columns[i + 1] != -1 &&
            (surplus[i] * surplus[dynamic_columns[i + 1]] < 0)) {
            // Found a candidate pair, so compute the number of atoms it can
            // shuffle and save this value in atoms_to_shuffle
            compressed_to_original[compressed_to_original_idx] = i;
            atoms_to_shuffle[compressed_to_original_idx] =
                min(abs(surplus[i]), abs(surplus[dynamic_columns[i + 1]]));

            // Compute absolute value difference in case a tiebreaker is needed
            // Smaller absolute value differences are preferred because they get
            // pairs of columns to completion quicker
            int max_abs_val =
                max(abs(surplus[i]), abs(surplus[dynamic_columns[i + 1]]));
            int min_abs_val =
                min(abs(surplus[i]), abs(surplus[dynamic_columns[i + 1]]));
            abs_val_diff[compressed_to_original_idx] =
                max_abs_val - min_abs_val;

            compressed_to_original_idx++;
        }
    }

    // Bubble sort from best to worst
    for (int i = 0; i < compressed_to_original_idx - 1; i++) {
        for (int j = 0; j < compressed_to_original_idx - i - 1; j++) {
            if ((atoms_to_shuffle[j] < atoms_to_shuffle[j + 1]) ||
                (atoms_to_shuffle[j] == atoms_to_shuffle[j + 1] &&
                 abs_val_diff[j] > abs_val_diff[j + 1])) {
                // Swap in all 3 arrays
                int temp = atoms_to_shuffle[j];
                atoms_to_shuffle[j] = atoms_to_shuffle[j + 1];
                atoms_to_shuffle[j + 1] = temp;

                temp = compressed_to_original[j];
                compressed_to_original[j] = compressed_to_original[j + 1];
                compressed_to_original[j + 1] = temp;

                temp = abs_val_diff[j];
                abs_val_diff[j] = abs_val_diff[j + 1];
                abs_val_diff[j + 1] = temp;
            }
        }
    }

    // Now that the arrays are sorted from best to worst, move the top
    // max_num_pairs_to_shuffle pairs to the donors and the receivers arrays

    // We need to keep track of the columns that have already been
    // matched/paired so as not to pick them twice
    int is_locked[W];
    for (int i = 0; i < W; i++) {
        is_locked[i] = 0;
    }

    // We may shuffle less than possible_num_pairs_to_shuffle in case
    // possible_num_pairs_to_shuffle is equal to compressed_to_original_idx and
    // the pairs are not mutually exclusive
    int possible_num_pairs_to_shuffle =
        min(compressed_to_original_idx, max_num_pairs_to_shuffle);
    int cur_compressed_idx = 0;
    int cur_output_index = 0;

    while (possible_num_pairs_to_shuffle &&
           (cur_compressed_idx < compressed_to_original_idx)) {
        int col1 = compressed_to_original[cur_compressed_idx];
        int col2 = dynamic_columns[col1 + 1];
        if (!is_locked[col1] && !is_locked[col2]) {
            is_locked[col1] = 1;
            is_locked[col2] = 1;
            if (surplus[col1] > 0) {
                donors[cur_output_index] = col1;
                receivers[cur_output_index] = col2;
            } else {
                donors[cur_output_index] = col2;
                receivers[cur_output_index] = col1;
            }
            cur_output_index++;
            possible_num_pairs_to_shuffle--;
        }
        cur_compressed_idx++;
    }

    *num_pairs_to_shuffle = cur_output_index;
    // Done!
}

// Check if the instance has been solved
// This does not check whether the intermediary steps have been
// executed properly; it only checks whether the final configuration
// covers the target region properly
int verify_redrec_solution(int *source, int num_source_atoms, int H, int W,
                           int R_h) {

    int num_atoms_in_grid = 0;
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            if (source[r * W + c] == 1) {
                num_atoms_in_grid += 1;
            }
        }
    }

    // Check if the number of atoms we ended up with is equal to
    // to the number of atoms we had initially
    if (num_atoms_in_grid != num_source_atoms) {
        return 0;
    }

    // Check if the target region is filled with 1s
    for (int r = R_h; r < H - R_h; r++) {
        for (int c = 0; c < W; c++) {
            if (source[r * W + c] != 1) {
                return 0;
            }
        }
    }

    return 1;
}

void sort_donor_and_shuffle(int *source, int H, int W, int R_h, int T_h,
                            int atoms_to_shuffle, int d_c, int r_c,
                            int *src_batched, int *dst_batched,
                            int *batchPtr_batched, int *num_batches_batched,
                            int *sol_length_batched) {
    // Keep track of the vacant traps in the top and the bottom reservoir
    int vacant_traps_in_top[R_h];
    int vacant_traps_in_top_idx = 0;
    int vacant_traps_in_bottom[R_h];
    int vacant_traps_in_bottom_idx = 0;

    // The two index variables above will indicate the number of vacant
    // traps in the top and the bottom reservoir as soon as we are done
    // populating the two arrays

    // Populate vacancies
    for (int r = 0; r < R_h; r++) {
        if (source[r * W + r_c] == 0) {
            vacant_traps_in_top[vacant_traps_in_top_idx++] = r;
        }
    }
    for (int r = H - R_h; r < H; r++) {
        if (source[r * W + r_c] == 0) {
            vacant_traps_in_bottom[vacant_traps_in_bottom_idx++] = r;
        }
    }

    // Populate the source array, which is nothing but the column of the donor
    int source_arr[H];

    // Set K and K_prime accordingly
    int K = 0;
    int K_prime = atoms_to_shuffle + T_h;

    for (int r = 0; r < H; r++) {
        source_arr[r] = source[r * W + d_c];
        if (source_arr[r] == 1) {
            K++;
        }
    }

    // Target can take on different forms, populate the center part first
    int target_arr[H];

    for (int r = 0; r < H; r++) {
        if (r < R_h || r >= H - R_h) {
            target_arr[r] = 0;
        } else {
            target_arr[r] = 1;
        }
    }

    // Now, we would like to select the split that minimizes displacement
    // A split is indicated by the number of atoms we decide to shuffle
    // to the top reservoir of the receiver
    int selected_split = -1;
    int min_displacement = INF;

    int num_vacancies_top = vacant_traps_in_top_idx;
    int num_vacancies_bottom = vacant_traps_in_bottom_idx;

    // We will use those variables to store results
    int src_batched_1d[H * H];
    int dst_batched_1d[H * H];
    int batchPtr_batched_1d[H];
    int num_batches_batched_1d;
    int sol_length_batched_1d;

    int atoms_from_top, atoms_from_bottom;

    for (atoms_from_top = 0; atoms_from_top <= atoms_to_shuffle;
         atoms_from_top++) {
        atoms_from_bottom = atoms_to_shuffle - atoms_from_top;
        // Check whether we have enough atoms on top & bottom for current split
        if (atoms_from_bottom <= num_vacancies_bottom &&
            atoms_from_top <= num_vacancies_top) {
            int cur_target[H];

            // Copy target to cur_target
            for (int i = 0; i < H; i++) {
                cur_target[i] = target_arr[i];
            }

            // Store which atoms we selected from top and bottom
            int atoms_selected_from_top[H];
            int atoms_selected_from_top_idx = 0;
            int atoms_selected_from_bottom[H];
            int atoms_selected_from_bottom_idx = 0;

            for (int i = num_vacancies_top - atoms_from_top;
                 i < num_vacancies_top; i++) {
                cur_target[vacant_traps_in_top[i]] = 1;
                atoms_selected_from_top[atoms_selected_from_top_idx++] =
                    vacant_traps_in_top[i];
            }

            for (int i = 0; i < atoms_from_bottom; i++) {
                cur_target[vacant_traps_in_bottom[i]] = 1;
                atoms_selected_from_bottom[atoms_selected_from_bottom_idx++] =
                    vacant_traps_in_bottom[i];
            }

            // Use the 1D Solver
            lin_exact_generate_moves_batched_cpu(
                source_arr, cur_target, H, K, K_prime, src_batched_1d,
                dst_batched_1d, batchPtr_batched_1d, &num_batches_batched_1d,
                &sol_length_batched_1d);

            // Displacement from the 1D Solver is equal to the length of the
            // solution
            int displacement = sol_length_batched_1d;

            int top_reservoir_receiver_displacement =
                get_top_reservoir_displacement(source, atoms_selected_from_top,
                                               atoms_selected_from_top_idx, R_h,
                                               H, W, r_c);
            int bottom_reservoir_receiver_displacement =
                get_bottom_reservoir_displacement(
                    source, atoms_selected_from_bottom,
                    atoms_selected_from_bottom_idx, R_h, H, W, r_c);

            // Compute total displacement
            int total_displacement = displacement +
                                     top_reservoir_receiver_displacement +
                                     bottom_reservoir_receiver_displacement;

            // Compare total displacement to the minimum we have so far
            if (total_displacement < min_displacement) {
                min_displacement = total_displacement;
                selected_split = atoms_from_top;
            }
        }
    }

    // Recreate the selected split
    atoms_from_top = selected_split;
    atoms_from_bottom = atoms_to_shuffle - atoms_from_top;

    int cur_target[H];

    for (int i = 0; i < H; i++) {
        cur_target[i] = target_arr[i];
    }

    int atoms_selected_from_top[H];
    int atoms_selected_from_top_idx = 0;
    int atoms_selected_from_bottom[H];
    int atoms_selected_from_bottom_idx = 0;

    for (int i = num_vacancies_top - atoms_from_top; i < num_vacancies_top;
         i++) {
        cur_target[vacant_traps_in_top[i]] = 1;
        atoms_selected_from_top[atoms_selected_from_top_idx++] =
            vacant_traps_in_top[i];
    }

    for (int i = 0; i < atoms_from_bottom; i++) {
        cur_target[vacant_traps_in_bottom[i]] = 1;
        atoms_selected_from_bottom[atoms_selected_from_bottom_idx++] =
            vacant_traps_in_bottom[i];
    }

    lin_exact_generate_moves_batched_cpu(
        source_arr, cur_target, H, K, K_prime, src_batched_1d, dst_batched_1d,
        batchPtr_batched_1d, &num_batches_batched_1d, &sol_length_batched_1d);

    convert_and_execute_1d_solver_output(
        source, W, d_c, src_batched_1d, dst_batched_1d, batchPtr_batched_1d,
        num_batches_batched_1d, sol_length_batched_1d, src_batched, dst_batched,
        batchPtr_batched, num_batches_batched, sol_length_batched);

    // To the above, we still have to add the reshuffling batches
    // The number of reshuffling batches is equal to the distance between the
    // two columns
    int distance_between_columns = abs(d_c - r_c);
    // Direction depends on the position of the donor with respect to the
    // receiver
    int direction = (r_c > d_c) ? 1 : -1;
    // The length of the solution here is known and is equal to the number of
    // atoms/ to be shuffled multiplied by the distance between the columns
    int reshuffling_src_batched[distance_between_columns * atoms_to_shuffle];
    int reshuffling_dst_batched[distance_between_columns * atoms_to_shuffle];
    // The number of batches is known too and is equal to the distance between
    // the columns
    int reshuffling_batchPtr_batched[distance_between_columns];
    // Despite the fact that those are known, we will use them as indices, so
    // they start at 0
    int reshuffling_num_batches_batched = 0;
    int reshuffling_sol_length_batched = 0;

    for (int i = 0; i < distance_between_columns; i++) {
        reshuffling_batchPtr_batched[reshuffling_num_batches_batched++] =
            reshuffling_sol_length_batched;
        int from_col = d_c + i * direction;
        int to_col = d_c + (i + 1) * direction;
        for (int j = 0; j < atoms_from_top; j++) {
            int atom_row = atoms_selected_from_top[j];
            reshuffling_src_batched[reshuffling_sol_length_batched] =
                coord_to_idx(atom_row, from_col, W);
            reshuffling_dst_batched[reshuffling_sol_length_batched] =
                coord_to_idx(atom_row, to_col, W);
            reshuffling_sol_length_batched++;
        }

        for (int j = 0; j < atoms_from_bottom; j++) {
            int atom_row = atoms_selected_from_bottom[j];
            reshuffling_src_batched[reshuffling_sol_length_batched] =
                coord_to_idx(atom_row, from_col, W);
            reshuffling_dst_batched[reshuffling_sol_length_batched] =
                coord_to_idx(atom_row, to_col, W);
            reshuffling_sol_length_batched++;
        }
    }

    // Finally, execute the moves above
    execute_shuffling_output(
        source, reshuffling_src_batched, reshuffling_dst_batched,
        reshuffling_batchPtr_batched, reshuffling_num_batches_batched,
        reshuffling_sol_length_batched, src_batched, dst_batched,
        batchPtr_batched, num_batches_batched, sol_length_batched);
}

float redrec_v2(int *initial, int initial_atom_count, int Nt_x, int Nt_y,
                int R_h, int *src, int *dst, int *batch_indices,
                int *num_batches, int *sol_length) {

    *sol_length = 0;
    *num_batches = 0;
    batching_time = 0;

    // Given the height of the reservoir, as well as the height of the grid,
    // we can easily calculate the height of the target region
    int T_h = Nt_y - 2 * R_h;

    // Keep track of the solved columns
    int column_solved[Nt_x];

    // Initially, all columns are unsolved
    for (int i = 0; i < Nt_x; i++) {
        column_solved[i] = 0;
    }

    // This array will keep track of the surplus within every column
    // The surplus is the number of atoms minus the number of targets
    int surplus[Nt_x];

    // Populate initial surplus
    compute_surplus(initial, Nt_y, Nt_x, T_h, surplus);

    // This array will allow us to keep track of columns after removal
    // We update this array using the initial columns and a suffix that
    // keeps track of the leftmost unsolved column for every column
    int dynamic_columns[Nt_x];

    // Initially, dynamic_columns is the columns in their regular order
    for (int i = 0; i < Nt_x; i++) {
        dynamic_columns[i] = i;
    }

    // For all columns where the deficit is bigger than 0 and is bigger than the
    // vacant spots in the reservoir, sort (everything goes to the middle)
    for (int c = 0; c < Nt_x; c++) {
        if (surplus[c] < 0 &&
            abs(surplus[c]) >
                get_vacant_reservoir_traps(initial, Nt_y, Nt_x, R_h, c)) {
            partial_sort(initial, c, surplus, T_h, Nt_y, Nt_x, src, dst,
                         batch_indices, num_batches, sol_length);
        }
    }

    // Sort columns that have a net surplus of zero
    sort_net_zero_columns(initial, Nt_y, Nt_x, T_h, R_h, surplus,
                          dynamic_columns, column_solved, src, dst,
                          batch_indices, num_batches, sol_length);

    while (deficit_exists(surplus, Nt_x)) {

        int d, r;

        // Find the next pair to shuffle atoms across
        // The criteria we are using to determine this pair can be changed later
        // on
        find_pair_to_shuffle(surplus, column_solved, dynamic_columns, &d, &r,
                             Nt_x);

        int donor_surplus = surplus[d];
        int receiver_deficit = abs(surplus[r]);

        sort_donor_and_shuffle(initial, Nt_y, Nt_x, R_h, T_h,
                               min(donor_surplus, receiver_deficit), d, r, src,
                               dst, batch_indices, num_batches, sol_length);

        if (donor_surplus == receiver_deficit) {
            // Donor is already taken care of; we need to sort the receiver and
            // mark it as solved

            // First, update the surpluses
            surplus[d] = 0;
            surplus[r] = 0;

            // Mark the donor as solved
            column_solved[d] = 1;
            // Update dynamic columns (which should be done every time a column
            // is marked as solved)
            update_dynamic_columns(Nt_x, column_solved, dynamic_columns);

            // Finally, solve the receiver
            sort_net_zero_column(initial, r, T_h, R_h, Nt_y, Nt_x,
                                 dynamic_columns, column_solved, src, dst,
                                 batch_indices, num_batches, sol_length);
        } else if (donor_surplus > receiver_deficit) {
            // Update the surpluses
            surplus[d] = donor_surplus - receiver_deficit;
            surplus[r] = 0;

            // Solve the receiver
            sort_net_zero_column(initial, r, T_h, R_h, Nt_y, Nt_x,
                                 dynamic_columns, column_solved, src, dst,
                                 batch_indices, num_batches, sol_length);
        } else {
            // Update the surpluses
            surplus[d] = 0;
            surplus[r] = donor_surplus - receiver_deficit;

            // Mark the donor as solved
            column_solved[d] = 1;
            // Update dynamic columns (which should be done every time a column
            // is marked as solved)
            update_dynamic_columns(Nt_x, column_solved, dynamic_columns);

            // Check if a partial sort is required for the receiver
            if (abs(surplus[r]) >
                get_vacant_reservoir_traps(initial, Nt_y, Nt_x, R_h, r)) {
                partial_sort(initial, r, surplus, T_h, Nt_y, Nt_x, src, dst,
                             batch_indices, num_batches, sol_length);
            }
        }
    }

    // We will be using those variables to store the results from the 1D solvers
    int src_batched_1d[Nt_y * Nt_y];
    int dst_batched_1d[Nt_y * Nt_y];
    int batchPtr_batched_1d[Nt_y];
    int num_batches_batched_1d;
    int sol_length_batched_1d;

    // Now that we are here, we know that the columns left are either already
    // solved or require solving, use 1D solver for those
    for (int c = 0; c < Nt_x; c++) {
        if (!column_solved[c]) {

            // Found a column that requires solving; since we know that
            // the target is centered, we will be using the bruteforce solver
            int source_arr[Nt_y];

            int K = 0;
            int K_prime = T_h;
            int K_1 = R_h;
            int K_2 = R_h;

            // Copy the column over
            for (int r = 0; r < Nt_y; r++) {
                source_arr[r] = initial[r * Nt_x + c];
                if (source_arr[r] == 1) {
                    K++;
                }
            }

            bruteforce_generate_moves_batched_cpu(
                source_arr, Nt_y, K, K_prime, K_1, K_2, src_batched_1d,
                dst_batched_1d, batchPtr_batched_1d, &num_batches_batched_1d,
                &sol_length_batched_1d);

            convert_and_execute_1d_solver_output(
                initial, Nt_x, c, src_batched_1d, dst_batched_1d,
                batchPtr_batched_1d, num_batches_batched_1d,
                sol_length_batched_1d, src, dst, batch_indices, num_batches,
                sol_length);
        }
    }

    return batching_time;
}
