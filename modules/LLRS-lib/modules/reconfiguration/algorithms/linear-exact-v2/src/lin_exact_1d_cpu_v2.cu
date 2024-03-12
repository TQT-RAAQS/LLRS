#include "lin_exact_1d_cpu_v2.h"
#include <stdlib.h>
#include <stdio.h>
void compute_heights_cpu(int* sourceFlags, int* targetFlags, int numTraps, int* height){
    int heightNow = 0;

    for(int i=0; i<numTraps; i++){
        if(sourceFlags[i]==1){
            heightNow++;
        }
    
        height[i] = heightNow;
    
        if(targetFlags[i]==1){
            heightNow--;
        }
    }
}

void get_primes_cpu(int* sourceFlags, int* targetFlags, int* yPrime, int numTraps, int numSources, int numTargets)
{
    int stackSources[numSources];
    int topSourcesStack = -1;
    int stackTargets[numTargets];
    int topTargetsStack = -1;

    for(int i=0; i<numTraps; i++){
        if(sourceFlags[i]==1){
            stackSources[++topSourcesStack] = i;
            if(topTargetsStack != -1){
                yPrime[stackTargets[topTargetsStack--]] = i;
            }
        }     
        if(targetFlags[i]==1){
            stackTargets[++topTargetsStack] = i;
            if(topSourcesStack != -1){
                yPrime[stackSources[topSourcesStack--]] = i;
            }
        } 
    }
}


void compute_profits_cpu(int* sourceFlags, int numExcessSources, int* height, int* max, int* max_index, int numTraps, int* yPrime, int* profit){
    
    for (int i=numTraps-1; i>-1; i--){
        if(sourceFlags[i]==1){
            if(height[i]>0 && height[i]<(numExcessSources+1)){
				int YDoublePrime=yPrime[yPrime[i]];
                profit[i] = 2*yPrime[i]-i-YDoublePrime+profit[YDoublePrime];
                
                if(profit[i] >= max[height[i]-1]){
                    max[height[i]-1] = profit[i];
                    max_index[height[i]-1] = i;
                }
            }
        }
    }
}

void lin_exact_cpu_v2_flags(int* inputSourceFlags, int* inputTargetFlags, int numTraps, int numSources, int numTargets, int* OutSourcesFlags){
    
	int numExcessSources = numSources - numTargets;
    int height[numTraps];    
    int yPrime[numTraps];
    int profit[numTraps];
	
	int sourceFlags[numTraps];
	int targetFlags[numTraps];
	
    for(int i=0; i<numTraps; i++){			
        yPrime[i] = numTraps-1;
        profit[i] = 0;
    }
    
	int source;
	int target;

    for(int i=0; i<numTraps; i++){
		source = inputSourceFlags[i];
		target  = inputTargetFlags[i];
        OutSourcesFlags[i] = source;
		
		if((source == 1) && (target == 1)){
			sourceFlags[i] = 0;
			targetFlags[i] = 0;
            numSources--;
            numTargets--;
		}
		else{
			sourceFlags[i] = source;
			targetFlags[i] = target;
		}
	}

    compute_heights_cpu(sourceFlags, targetFlags, numTraps, height);
	
    get_primes_cpu(sourceFlags, targetFlags, yPrime, numTraps, numSources, numTargets);

    int max_index[numExcessSources];
    int max[numExcessSources];
    for(int i=0; i<numExcessSources; i++){
        max[i] = - __INT_MAX__;
    }
	
	compute_profits_cpu(sourceFlags,numExcessSources,height,max,max_index,numTraps, yPrime, profit);

	for(int i=0;i<numExcessSources;i++){
        OutSourcesFlags[max_index[i]] = 0;	
    }

}

void lin_exact_cpu_v2_generate_matching(int* sourceFlags, int* targetFlags, int numTraps, int numSources, int numTargets, int* OutSources_cpu, int* OutTargets_cpu){   

    int numExcessSources = numSources-numTargets;
    int countTargets = 0;

    for(int i=0; i<numTraps; i++){
        if(targetFlags[i] == 1){
            OutTargets_cpu[countTargets] = i;
			countTargets++;
        }
    }

    if(numExcessSources<0){
        for(int i=0; i<numTargets; i++){
            OutSources_cpu[i] = -1;
        }
        return;
    }

    if (numExcessSources==0){
        int countSources = 0;
        for (int i=0; i<numTraps; i++){
            if(sourceFlags[i] == 1){
                OutSources_cpu[countSources++] = i;
            }
        }
        return;
    }

    int OutSourcesFlags[numTraps];

    lin_exact_cpu_v2_flags(sourceFlags, targetFlags, numTraps, numSources, numTargets, OutSourcesFlags);

    int countSources = 0;

    for (int i=0; i<numTraps; i++){
        if(OutSourcesFlags[i] == 1){
            OutSources_cpu[countSources++] = i;
        }
    }

}

void generate_output_for_blocks_going_left_v2(int matching_going_left_src[], int matching_going_left_dst[], int mgl_idx, int target_atom_count, int* src, int* dst, int* blk_sizes, int* batch_indices, int* num_batches, int* sol_length) {
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

void generate_output_for_blocks_going_right_v2(int matching_going_right_src[], int matching_going_right_dst[], int mgr_idx, int target_atom_count, int* src, int* dst, int* blk_sizes, int* batch_indices, int* num_batches, int* sol_length) {

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

void lin_exact_1d_cpu_v2_block_output_generator(int matching_src[], int matching_dst[], int target_atom_count, int* src, int* dst, int* blk_sizes, int* batch_indices, int* num_batches, int* sol_length) {

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

    generate_output_for_blocks_going_right_v2(matching_going_right_src, matching_going_right_dst, mgr_idx, target_atom_count, src, dst, blk_sizes, batch_indices, num_batches, sol_length);
    generate_output_for_blocks_going_left_v2(matching_going_left_src, matching_going_left_dst, mgl_idx, target_atom_count, src, dst, blk_sizes, batch_indices, num_batches, sol_length);
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

    for(int u = 0 ; u < K_prime; u++){
        printf("%d,", matching_src[u]);
    }
    printf("\n");
    for(int u = 0 ; u < K_prime; u++){
        printf("%d,", matching_dst[u]);
    }

    generate_output_for_blocks_going_right_v2(matching_going_right_src, matching_going_right_dst, mgr_idx, K_prime, src_block_inv, dst_block_inv, size_block_inv, batchPtr_block_inv, &num_batches_block_inv, &sol_length_block_inv);
    generate_output_for_blocks_going_left_v2(matching_going_left_src, matching_going_left_dst, mgl_idx, K_prime, src_block_inv, dst_block_inv, size_block_inv, batchPtr_block_inv, &num_batches_block_inv, &sol_length_block_inv);

    int deref_solution_l = sol_length_block_inv;
    //printf("%d\n", deref_solution_l);

    // For things going left before the inversion, make sure to use the leftmost atom as the head
    // for (int i = 0; i < sol_length_block_inv; i++) {
    //     if (src_block_inv[i] > dst_block_inv[i]) {
    //         // Offset both by the size of the block
    //         src_block_inv[i] -= size_block_inv[i] - 1;
    //         dst_block_inv[i] -= size_block_inv[i] - 1;
    //     }
    // }

    // printf("Before inversion \n");
    // for(int p = 0; p < deref_solution_l; p++){

    //     printf("%d,", src_block_inv[p]);

    // }
    // printf("\n");
    // for(int p = 0; p < deref_solution_l; p++){

    //     printf("%d,", dst_block_inv[p]);

    // }
    // printf("\n");
    // for(int p = 0; p < deref_solution_l; p++){

    //     printf("%d,", size_block_inv[p]);

    // }
    // printf("\n");

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
            // printf("inverting:\n");
            // printf("src_block[%d] = %d\n", output_idx, new_src);
            // printf("dst_block[%d] = %d\n", output_idx, new_dst);
            // printf("size_block[%d] = %d\n", output_idx, new_size_block);
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

void lin_exact_cpu_v2_block_batched(int* initial, int* target, int num_traps, int inital_atom_count, int target_atom_count, int* src, int* dst, int* blk_sizes, int* batch_indices, int* num_batches, int* sol_length) {

    int matching_src[target_atom_count];
    int matching_dst[target_atom_count];

    lin_exact_cpu_v2_generate_matching(initial, target, num_traps, inital_atom_count, target_atom_count, matching_src, matching_dst);

    lin_exact_1d_cpu_v2_block_output_generator(matching_src, matching_dst, target_atom_count, src, dst, blk_sizes, batch_indices, num_batches, sol_length);
    
}

void lin_exact_cpu_v2_clump_batched(int* source, int* target, int N, int K, int K_prime, int* src_c_block, int* dst_c_block, int* size_c_block, int* batchPtr_c_block, int* num_batches_c_block, int* sol_length_c_block) {
   
    int matching_src[K_prime];
    int matching_dst[K_prime];

    lin_exact_cpu_v2_generate_matching(source, target, N, K, K_prime, matching_src, matching_dst);

    // Do the flip here
    clumped_block_output_generator_cpu(matching_dst, matching_src, N, K_prime, src_c_block, dst_c_block, size_c_block, batchPtr_c_block, num_batches_c_block, sol_length_c_block);
}

void moves_generator_unbatched_cpu(int* OutSources, int* OutTargets, int numTargets, int* outputMovesSource, int* outputMovesTarget, int * sol_length) {

    int currentIndex = 0;
    
    for (int i = numTargets - 1; i >= 0; i--) {
        while (OutSources[i] < OutTargets[i]) {
            outputMovesSource[currentIndex] = OutSources[i];
            outputMovesTarget[currentIndex++] = OutSources[i] + 1;
            OutSources[i]++;
            *sol_length++;
        }
    }    

    for (int i = 0; i < numTargets; i++) {
        while (OutSources[i] > OutTargets[i]) {
            outputMovesSource[currentIndex] = OutSources[i];
            outputMovesTarget[currentIndex++] = OutSources[i] - 1;
            OutSources[i]--;
            *sol_length++;
        }
    }
}

void lin_exact_cpu_v2_unbatched_moves(int* source, int* target, int N, int K, int K_prime, int * src_moves, int * dst_moves, int* sol_length) {

    int matching_src[K_prime]; 
    int matching_dst[K_prime];
    
    lin_exact_cpu_v2_generate_matching(source, target, N, K, K_prime, matching_src, matching_dst);

    moves_generator_unbatched_cpu(matching_src, matching_dst, K_prime, src_moves, dst_moves, sol_length);

}

