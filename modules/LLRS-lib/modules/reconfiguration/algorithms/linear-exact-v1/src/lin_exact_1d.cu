#define INF ~(1 << 31)/2

int min(int a, int b) {
    if (a < b) return a;
    return b;
}

int max(int a, int b) {
    if (a < b) return b;
    return a;
}

int get_excess(int* initial, int* target, int num_traps) {
    int excess = 0;
    for (int i = 1; i <= num_traps; i++) {
        if (initial[i]) excess++;
        if (target[i]) excess--;
    }
    return excess;
}

void compute_heights(int* initial, int* target, int num_traps, int heights[], int* excess) {

    int cur_num_sources_inclusive = 0;
    int cur_num_target_exclusive = 0;

    heights[0] = 0;

    for (int i = 1; i <= num_traps; i++) {
        cur_num_sources_inclusive += initial[i];
        cur_num_target_exclusive += target[i - 1];
        heights[i] = cur_num_sources_inclusive - cur_num_target_exclusive;
    }

    heights[num_traps + 1] = (*excess) = get_excess(initial, target, num_traps);
}


// matching generator
void lin_exact_generate_matching(int* initial, int* target, int num_traps, int initial_atom_count, int target_atom_count, int matching_src[], int matching_dst[]) {
    int X = num_traps + 1;
    // Including 0; heights[0] = 0 and heights[num_traps + 1] = excess
    int heights[num_traps + 2];

    // Offset the initial just so that the sources and targets are in (0, X) (set X to num_traps + 1)
    int source_with_offset[num_traps+1];
    int target_with_offset[num_traps+1];

    source_with_offset[0] = 0;
    target_with_offset[0] = 0;
    for (int i = 1; i <= num_traps; i++) {
        source_with_offset[i] = initial[i-1];
        target_with_offset[i] = target[i-1];
    }

    // Compress initial and target
    int S_set[initial_atom_count];
    int S_set_idx;
    S_set_idx = 0;

    for (int i = 1; i <= num_traps; i++) {
        if (source_with_offset[i]) {
            S_set[S_set_idx++] = i;
        }
    }

    int excess;

    compute_heights(source_with_offset, target_with_offset, num_traps, heights, &excess);

    int h_max, h_min;

    h_max = h_min = heights[0];

    for (int i = 0; i <= X; i++) {
        h_max = max(h_max, heights[i]);
        h_min = min(h_min, heights[i]);
    }

    // Those are only defined for [h_min...h_max] so make sure to index properly
    // The way this will work is that S[k] will be an array of indices
    int S_tilde[h_max - h_min + 1][num_traps];
    int T_tilde[h_max - h_min + 1][num_traps];
    int S_tilde_indices[h_max - h_min + 1];
    int T_tilde_indices[h_max - h_min + 1];
    // We can get the corresponding k using heights, the below arrays store the indices within the k
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

    for (int i = 1; i <= num_traps; i++) {
        if (source_with_offset[i]) {
            S_tilde[heights[i] - h_min][S_tilde_indices[heights[i] - h_min]] = i;
            // Map i to S_tilde index
            source_to_S_tilde_index[i] = S_tilde_indices[heights[i] - h_min];
            // Increment indices
            S_tilde_indices[heights[i] - h_min]++;
        }
        if (target_with_offset[i]) {
            T_tilde[heights[i] - h_min][T_tilde_indices[heights[i] - h_min]] = i;
            // Increment indices
            T_tilde_indices[heights[i] - h_min]++;
        }
    }

    // Add the X where necessary
    for (int k = h_min; k <= h_max; k++) {
        if (k <= excess) {
            T_tilde[k - h_min][T_tilde_indices[k - h_min]++] = X;
        }
        // This is for the corner case that we had to take care of, may or may not work
        else {
            S_tilde[k - h_min][S_tilde_indices[k - h_min]++] = X;
        }
    }
    
    int P[X + 1];

    P[X] = 0;

    // Loop over the elements in S
    for (int i = initial_atom_count - 1; i >= 0; i--) {
        // Get y
        int y = S_set[i];
        // Get height of y
        int height = heights[y];
        // Get adjusted height
        int k = height - h_min;
        // We are looking for y', which is defined for a y in S_tilde as whatever is in T_tilde at the same index
        int y_prime;
        if (y != X) {
            y_prime = T_tilde[k][source_to_S_tilde_index[y]];
        }
        else {
            // Corner case: X' = X (that will never happen here because S_set[i] will never be equal to X, but it may happen for X'')
            y_prime = X;
        }
        // We are also looking for y'', which is defined for a y' in T_tilde as whatever is in S_tilde at the next index
        int y_double_prime;
        if (y_prime != X) {
            y_double_prime = S_tilde[k][source_to_S_tilde_index[y] + 1];
        } else {
            // Corner case: X' = X
            y_double_prime = X;
        }
        // Finally, compute P
        P[y] = 2*y_prime - y - y_double_prime + P[y_double_prime]; 
    }

    // pi is not defined for h_min but for ease of indexing we will assume it is
    int pi[h_max - h_min + 1];
    // Initialize to -INF
    for (int i = 0; i < h_max - h_min + 1; i++) {
        pi[i] = -INF;
    }

    // Go over all the Ps and find the max of every pi[k]
    for (int i = 0; i < initial_atom_count; i++) {
        int y = S_set[i];
        int k = heights[y];
        pi[k - h_min] = max(pi[k - h_min], P[y]);
    }

    int to_be_removed[num_traps + 1];
    int max_elem[h_max - h_min + 1];
    int max_value[h_max - h_min + 1];

    for (int i = 1; i <= num_traps; i++) {
        to_be_removed[i] = 0;
    }

    for (int i = 0; i < h_max - h_min + 1; i++) {
        max_elem[i] = -1;
        max_value[i] = -INF;
    }

    // pi contains the maximum profit needed for heights between 1 and excess (inclusive), so we just loop 
    // over the values now!
    for (int i = 0; i < initial_atom_count; i++) {
        int y = S_set[i];
        if (P[y] > max_value[heights[y] - h_min]) {
            max_elem[heights[y] - h_min] = y;
            max_value[heights[y] - h_min] = P[y];
        }
    }

    // Populate to_be_removed
    for (int i = 1; i <= excess; i++) {
        to_be_removed[max_elem[i - h_min]] = 1;
    }

    int matching_src_index = 0;
    int matching_dst_index = 0;

    // Write to output, do not forget to offset back to make things 0-indexed again
    for (int i = 1; i <= num_traps; i++) {
        if (source_with_offset[i] && !to_be_removed[i]) {
            matching_src[matching_src_index++] = i - 1;
        }
        if (target_with_offset[i]) {
            matching_dst[matching_dst_index++] = i - 1;
        }
    }
}

void generate_output_for_blocks_going_left(int matching_going_left_src[], int matching_going_left_dst[], int mgl_idx, int target_atom_count, int* src, int* dst, int* blk_sizes, int* batch_indices, int* num_batches, int* sol_length) {
    // Figure out the heads, a head is the rightmost atom in a block going left
    int is_head[mgl_idx];
    int block_to_head[mgl_idx];
    int block_to_tail[mgl_idx];

    int block_to_head_index = 0;
    int block_to_tail_index = 0;

    for (int i = 0; i < mgl_idx; i++) {
        if ((i == mgl_idx - 1) || !(matching_going_left_src[i] == matching_going_left_src[i + 1] - 1)) {
            is_head[i] = 1;
            block_to_head[block_to_head_index++] = i;
        }
        else {
            is_head[i] = 0;
        }
    }

    // Do something similar for tails
    for (int i = 0; i < mgl_idx; i++) {
        if ((i == 0) || !(matching_going_left_src[i] == matching_going_left_src[i - 1] + 1)) {
            block_to_tail[block_to_tail_index++] = i;
        }
    }

    // Now that we have the heads, do prefix sum to figure out the blocks for each index
    int index_to_block[mgl_idx];

    index_to_block[0] = 0;

    // For blocks going left, we will need exclusive prefix sum
    for (int i = 1; i < mgl_idx; i++) {
        // A head is the end of a block
        index_to_block[i] = index_to_block[i-1] + is_head[i - 1];
    }

    // The number of batches going left is equal to the maximum distance going left
    int max_distance_of_atoms_going_left = 0;

    for (int i = 0; i < mgl_idx; i++) {
        max_distance_of_atoms_going_left = max(max_distance_of_atoms_going_left, matching_going_left_src[i] - matching_going_left_dst[i]);
    }

    // Offset
    int num_batches_going_right = *num_batches;
    
    // Update the total number of batches
    *num_batches += max_distance_of_atoms_going_left;

    for (int j = num_batches_going_right; j < *num_batches; j++) {
        batch_indices[j] = *sol_length;

        for (int i = 0; i < mgl_idx; i++) {
            // Get block of atom
            int cur_block = index_to_block[i];
            if (block_to_head[cur_block] == i && matching_going_left_src[i] != matching_going_left_dst[i]) {
                // Head found, write to output
                blk_sizes[*sol_length] = i - block_to_tail[cur_block] + 1;
                src[*sol_length] = matching_going_left_src[i] - blk_sizes[*sol_length] + 1;
                dst[*sol_length] = matching_going_left_src[i] - blk_sizes[*sol_length];
                (*sol_length)++;
            }
            if (matching_going_left_src[i] != matching_going_left_dst[i]) {
                // Move once
                matching_going_left_src[i] -= 1;
            }
        }

        // Update heads
        for (int i = 0; i < mgl_idx - 1; i++) {
            if (matching_going_left_src[i] != matching_going_left_dst[i] && matching_going_left_src[i + 1] == matching_going_left_dst[i + 1] && index_to_block[i] == index_to_block[i+1]) {
                // Update head
                block_to_head[index_to_block[i]] = i;
            }
        }
    }
}

void generate_output_for_blocks_going_right(int matching_going_right_src[], int matching_going_right_dst[], int mgr_idx, int target_atom_count, int* src, int* dst, int* blk_sizes, int* batch_indices, int* num_batches, int* sol_length) {

    // Figure out the heads, a head is the leftmost atom in a block going right
    int is_head[mgr_idx];
    int block_to_head[mgr_idx];
    int block_to_tail[mgr_idx];

    int block_to_head_index = 0;
    int block_to_tail_index = 0;

    for (int i = 0; i < mgr_idx; i++) {
        if ((i == 0) || !(matching_going_right_src[i] == matching_going_right_src[i-1] + 1)) {
            is_head[i] = 1;
            block_to_head[block_to_head_index++] = i;
        }
        else {
            is_head[i] = 0;
        }
    }

    // Do something similar for tails
    for (int i = 0; i < mgr_idx; i++) {
        if ((i == mgr_idx - 1) || !(matching_going_right_src[i] == matching_going_right_src[i+1] - 1)) {
            block_to_tail[block_to_tail_index++] = i;
        }
    }

    // Now that we have the heads, do prefix sum to figure out the blocks for each index
    int index_to_block[mgr_idx];

    index_to_block[0] = 0;

    for (int i = 1; i < mgr_idx; i++) {
        // A head is the beginning of a new block
        index_to_block[i] = index_to_block[i-1] + is_head[i];
    }

    // The number of batches going right is equal to the maximum distance going right
    int max_distance_of_atoms_going_right = 0;

    for (int i = 0; i < mgr_idx; i++) {
        max_distance_of_atoms_going_right = max(max_distance_of_atoms_going_right, matching_going_right_dst[i] - matching_going_right_src[i]);
    }
    
    // Update the total number of batches
    *num_batches += max_distance_of_atoms_going_right;

    for (int j = 0; j < max_distance_of_atoms_going_right; j++) {
        batch_indices[j] = *sol_length;

        for (int i = 0; i < mgr_idx; i++) {
            // Get block of atom
            int cur_block = index_to_block[i];
            if (block_to_head[cur_block] == i && matching_going_right_src[i] != matching_going_right_dst[i]) {
                // Head found, write to output
                blk_sizes[*sol_length] = block_to_tail[cur_block] - i + 1;
                src[*sol_length] = matching_going_right_src[i];
                dst[*sol_length] = matching_going_right_src[i] + 1;
                (*sol_length)++;
            }
            if (matching_going_right_src[i] != matching_going_right_dst[i]) {
                // Move once
                matching_going_right_src[i] += 1;
            }
        }

        // Update heads
        for (int i = 1; i < mgr_idx; i++) {
            if (matching_going_right_src[i] != matching_going_right_dst[i] && matching_going_right_src[i - 1] == matching_going_right_dst[i - 1] && index_to_block[i] == index_to_block[i-1]) {
                // Update head
                block_to_head[index_to_block[i]] = i;
            }
        }
    }
}

void block_output_generator_cpu(int matching_src[], int matching_dst[], int target_atom_count, int* src, int* dst, int* blk_sizes, int* batch_indices, int* num_batches, int* sol_length) {

    *sol_length = 0;
    *num_batches = 0;

    // Split up the matching into two parts: things going right and things going left
    int matching_going_right_src[target_atom_count];
    int matching_going_right_dst[target_atom_count];
    int matching_going_left_src[target_atom_count];
    int matching_going_left_dst[target_atom_count];

    int mgr_idx = 0;
    int mgl_idx = 0;

    for (int i = 0; i < target_atom_count; i++) {
        if (matching_src[i] < matching_dst[i]) {
            matching_going_right_src[mgr_idx] = matching_src[i];
            matching_going_right_dst[mgr_idx] = matching_dst[i];
            mgr_idx++;
        }
        else if (matching_src[i] > matching_dst[i]) {
            matching_going_left_src[mgl_idx] = matching_src[i];
            matching_going_left_dst[mgl_idx] = matching_dst[i];
            mgl_idx++;
        }
    }

    generate_output_for_blocks_going_right(matching_going_right_src, matching_going_right_dst, mgr_idx, target_atom_count, src, dst, blk_sizes, batch_indices, num_batches, sol_length);
    generate_output_for_blocks_going_left(matching_going_left_src, matching_going_left_dst, mgl_idx, target_atom_count, src, dst, blk_sizes, batch_indices, num_batches, sol_length);
}

void lin_exact_block_batched_cpu(int* initial, int* target, int num_traps, int inital_atom_count, int target_atom_count, int* src, int* dst, int* blk_sizes, int* batch_indices, int* num_batches, int* sol_length) {

    int matching_src[target_atom_count];
    int matching_dst[target_atom_count];

    lin_exact_generate_matching(initial, target, num_traps, inital_atom_count, target_atom_count, matching_src, matching_dst);

    block_output_generator_cpu(matching_src, matching_dst, target_atom_count, src, dst, blk_sizes, batch_indices, num_batches, sol_length);
    
}

void clumped_block_output_generator_cpu(int matching_src[], int matching_dst[], int N, int K_prime, int* src_block, int* dst_block, int* size_block, int* batchPtr_block, int* num_batches_block, int* sol_length_block) {

    *sol_length_block = 0;
    *num_batches_block = 0;

    // Split up the matching into two parts: things going right and things going left
    int matching_going_right_src[K_prime];
    int matching_going_right_dst[K_prime];
    int matching_going_left_src[K_prime];
    int matching_going_left_dst[K_prime];

    int mgr_idx = 0;
    int mgl_idx = 0;

    for (int i = 0; i < K_prime; i++) {
        if (matching_src[i] < matching_dst[i]) {
            matching_going_right_src[mgr_idx] = matching_src[i];
            matching_going_right_dst[mgr_idx] = matching_dst[i];
            mgr_idx++;
        }
        else if (matching_src[i] > matching_dst[i]) {
            matching_going_left_src[mgl_idx] = matching_src[i];
            matching_going_left_dst[mgl_idx] = matching_dst[i];
            mgl_idx++;
        }
    }

    // Generate the output by treating the target as the source, and the source as the target
    int* src_block_inv = (int*)malloc((N*N)*sizeof(int));
    int* dst_block_inv = (int*)malloc((N*N)*sizeof(int));
    int* size_block_inv = (int*)malloc((N*N)*sizeof(int));
    int* batchPtr_block_inv = (int*)malloc(N*sizeof(int));
    int num_batches_block_inv = 0;
    int sol_length_block_inv = 0;

    generate_output_for_blocks_going_right(matching_going_right_src, matching_going_right_dst, mgr_idx, K_prime, src_block_inv, dst_block_inv, size_block_inv, batchPtr_block_inv, &num_batches_block_inv, &sol_length_block_inv);
    generate_output_for_blocks_going_left(matching_going_left_src, matching_going_left_dst, mgl_idx, K_prime, src_block_inv, dst_block_inv, size_block_inv, batchPtr_block_inv, &num_batches_block_inv, &sol_length_block_inv);

    // For things going left before the inversion, make sure to use the leftmost atom as the head
    for (int i = 0; i < sol_length_block_inv; i++) {
        if (src_block_inv[i] > dst_block_inv[i]) {
            // Offset both by the size of the block
            src_block_inv[i] -= size_block_inv[i] - 1;
            dst_block_inv[i] -= size_block_inv[i] - 1;
        }
    }

    // Now, output the actual output
    *num_batches_block = num_batches_block_inv;
    *sol_length_block = sol_length_block_inv;

    int batchptr_idx = 0;
    int output_idx = 0;
    for (int i = num_batches_block_inv - 1; i >= 0; i--) {
        // Get where the batch starts
        int batch_start = batchPtr_block_inv[i];
        // Get where the batch ends
        int batch_end = (i == num_batches_block_inv - 1) ? sol_length_block_inv : batchPtr_block_inv[i + 1];
        for (int j = batch_start; j < batch_end; j++) {
            // Get old src
            int old_src = src_block_inv[j];
            // Get old dst
            int old_dst = dst_block_inv[j];
            // Get old block size
            int old_size_block = size_block_inv[j];
            // Get old direction
            int direction = (old_src > old_dst) ? -1 : 1;
            // Get new src
            int new_src = old_dst;
            // Get new dst 
            int new_dst = old_src;
            // Get new block size
            int new_size_block = old_size_block;

            // Write to output
            src_block[output_idx] = new_src;
            dst_block[output_idx] = new_dst;
            size_block[output_idx] = new_size_block;
            output_idx++;
        } 
        batchPtr_block[num_batches_block_inv - 1 - i] = batchptr_idx;
        batchptr_idx += batch_end - batch_start;
    }

    free(batchPtr_block_inv);
    free(size_block_inv);
    free(dst_block_inv);
    free(src_block_inv);
}

void lin_exact_clump_batched_cpu(int* source, int* target, int N, int K, int K_prime, int* src_c_block, int* dst_c_block, int* size_c_block, int* batchPtr_c_block, int* num_batches_c_block, int* sol_length_c_block) {
   
    int matching_src[K_prime];
    int matching_dst[K_prime];

    lin_exact_generate_matching(source, target, N, K, K_prime, matching_src, matching_dst);

    // Do the flip here
    clumped_block_output_generator_cpu(matching_dst, matching_src, N, K_prime, src_c_block, dst_c_block, size_c_block, batchPtr_c_block, num_batches_c_block, sol_length_c_block);
}
