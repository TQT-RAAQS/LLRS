#include "lin_exact_1d_cpu_v2.h"
#include "redrec_cpu_v3.h"
#include <stdio.h>
#define moves_gen

void moves_generator_unbatched_column(int width, int *OutSources,
                                      int *OutTargets, int numTargets,
                                      int *outputMovesSource,
                                      int *outputMovesTarget,
                                      int *moves_counter, int *offset) {

    int currentIndex = *moves_counter;
    int colOffset = *offset;
    for (int i = numTargets - 1 + colOffset; i >= colOffset; i--) {
        int sourceRow = OutSources[i] / width;
        int sourceColumn = OutSources[i] % width;
        int destRow = OutTargets[i] / width;
        int destColumn = OutTargets[i] % width;

        while (sourceColumn < destColumn) {

            outputMovesSource[currentIndex] = OutSources[i];
            outputMovesTarget[currentIndex++] = OutSources[i] + 1;
            OutSources[i]++;
            (*moves_counter)++;
            sourceColumn = OutSources[i] % width;
        }
        while (sourceColumn > destColumn) {

            outputMovesSource[currentIndex] = OutSources[i];
            outputMovesTarget[currentIndex++] = OutSources[i] - 1;
            OutSources[i]--;
            (*moves_counter)++;
            sourceColumn = OutSources[i] % width;
        }
        while (sourceRow < destRow) {

            outputMovesSource[currentIndex] = OutSources[i];
            outputMovesTarget[currentIndex++] = OutSources[i] + width;
            OutSources[i] += width;
            (*moves_counter)++;
            sourceRow = OutSources[i] / width;
        }
    }

    for (int i = colOffset; i < numTargets + colOffset; i++) {
        int sourceRow = OutSources[i] / width;
        int sourceColumn = OutSources[i] % width;
        int destRow = OutTargets[i] / width;
        int destColumn = OutTargets[i] % width;

        while (sourceColumn < destColumn) {

            outputMovesSource[currentIndex] = OutSources[i];
            outputMovesTarget[currentIndex++] = OutSources[i] + 1;
            OutSources[i]++;
            (*moves_counter)++;
            sourceColumn = OutSources[i] % width;
        }
        while (sourceColumn > destColumn) {

            outputMovesSource[currentIndex] = OutSources[i];
            outputMovesTarget[currentIndex++] = OutSources[i] - 1;
            OutSources[i]--;
            (*moves_counter)++;
            sourceColumn = OutSources[i] % width;
        }

        while (sourceRow > destRow) {

            outputMovesSource[currentIndex] = OutSources[i];
            outputMovesTarget[currentIndex++] = OutSources[i] - width;
            OutSources[i] -= width;
            (*moves_counter)++;
            sourceRow = OutSources[i] / width;
        }
    }
    (*offset) += numTargets;
}

void update_dynamic_columns(int width, int *column_solved,
                            int *dynamic_columns) {
    int leftmost_unsolved_column = -1;
    for (int c = width - 1; c >= 0; c--) {
        if (!column_solved[c]) {
            leftmost_unsolved_column = c;
        }
        dynamic_columns[c] = leftmost_unsolved_column;
    }
}

int deficit_exists(int *surplus, int width) {
    for (int col = 0; col < width; col++) {
        if (surplus[col] < 0) {
            return 1;
        }
    }
    return 0;
}

void linear_solve(int column, int gridHeight, int reservoirHeight, int width,
                  int *sourceFlags, int *OutSources_cpu, int *OutTargets_cpu,
                  int *currentIndex, int targetIndex) {

    for (int row = 0; row < gridHeight; row++) {
        if (sourceFlags[row * width + column] == 1) {
            OutSources_cpu[*currentIndex] = row * width + column;
            OutTargets_cpu[*currentIndex] = targetIndex * width + column;
            (*currentIndex)++;
            targetIndex++;
        }
    }
}

void append_up_sources(int *ToMove, int *OutSources_cpu, int *OutTargets_cpu,
                       int *currentIndex, int reservoirHeight, int width,
                       int donor, int receiver, int *targetIndex) {
    for (int i = 0; i < reservoirHeight; i++) {
        if (ToMove[i] == 1) {
            OutSources_cpu[*currentIndex] = i * width + donor;
            OutTargets_cpu[*currentIndex] = *targetIndex * width + receiver;
            (*currentIndex)++;
            (*targetIndex)++;
        }
    }
}

void append_down_sources(int *ToMove, int *OutSources_cpu, int *OutTargets_cpu,
                         int *currentIndex, int gridHeight, int reservoirHeight,
                         int width, int donor, int receiver, int *targetIndex) {

    for (int i = gridHeight - reservoirHeight; i < gridHeight; i++) {
        if (ToMove[i] == 1) {
            OutSources_cpu[*currentIndex] = i * width + donor;
            OutTargets_cpu[*currentIndex] = *targetIndex * width + receiver;
            (*currentIndex)++;
            (*targetIndex)++;
        }
    }
}

void set_sources(int reservoirHeight, int gridHeight, int width,
                 int *sourceFlags, int *LocalsourceFlags, int donor,
                 int *numSources) {

    for (int i = 0; i < gridHeight; i++) {
        if ((i < reservoirHeight) || (i >= (gridHeight - reservoirHeight))) {
            if (sourceFlags[i * width + donor] == 1) {
                LocalsourceFlags[i] = 1;
                (*numSources)++;
            } else {
                LocalsourceFlags[i] = 0;
            }
        } else {
            LocalsourceFlags[i] = 0;
        }
    }
}

void find_pair_to_shuffle(int *surplus, int *column_solved,
                          int *dynamic_columns, int *donor, int *receiver,
                          int width) {
    int maxAtomsToShuffle = -__INT_MAX__;
    int atomsToShuffle;
    int min_dist;
    int dist;
    int maxAbsVal;
    int minAbsVal;
    int absValDiff;
    int selectedPairIndex = -1;
    for (int i = 0; i < width - 1; i++) {
        if (!column_solved[i] && dynamic_columns[i + 1] != -1 &&
            (surplus[i] * surplus[dynamic_columns[i + 1]] < 0)) {
            atomsToShuffle =
                min(abs(surplus[i]), abs(surplus[dynamic_columns[i + 1]]));
            if (atomsToShuffle > maxAtomsToShuffle) {
                maxAtomsToShuffle = atomsToShuffle;
                min_dist = dynamic_columns[i + 1] - i;
                maxAbsVal =
                    max(abs(surplus[i]), abs(surplus[dynamic_columns[i + 1]]));
                minAbsVal =
                    min(abs(surplus[i]), abs(surplus[dynamic_columns[i + 1]]));
                absValDiff = maxAbsVal - minAbsVal;
                selectedPairIndex = i;
            } else if (atomsToShuffle == maxAtomsToShuffle) {
                dist = dynamic_columns[i + 1] - i;
                if (dist < min_dist) {
                    min_dist = dist;
                    maxAbsVal = max(abs(surplus[i]),
                                    abs(surplus[dynamic_columns[i + 1]]));
                    minAbsVal = min(abs(surplus[i]),
                                    abs(surplus[dynamic_columns[i + 1]]));
                    absValDiff = maxAbsVal - minAbsVal;
                    selectedPairIndex = i;
                } else if (dist == min_dist) {
                    maxAbsVal = max(abs(surplus[i]),
                                    abs(surplus[dynamic_columns[i + 1]]));
                    minAbsVal = min(abs(surplus[i]),
                                    abs(surplus[dynamic_columns[i + 1]]));
                    if (maxAbsVal - minAbsVal < absValDiff) {
                        absValDiff = maxAbsVal - minAbsVal;
                        selectedPairIndex = i;
                    }
                }
            }
        }
    }
    if (surplus[selectedPairIndex] > 0) {
        *donor = selectedPairIndex;
        *receiver = dynamic_columns[selectedPairIndex + 1];
    } else {
        *donor = dynamic_columns[selectedPairIndex + 1];
        *receiver = selectedPairIndex;
    }
}
void count_up_sources(int reservoirHeight, int width, int *sourceFlags,
                      int donor, int *sourcesUp) {
    for (int i = 0; i < reservoirHeight; i++) {
        if (sourceFlags[i * width + donor] == 1) {
            (*sourcesUp) += 1;
        }
    }
}
void count_down_sources(int gridHeight, int targetHeight, int width,
                        int *sourceFlags, int donor, int *sourcesDown) {
    for (int i = gridHeight - targetHeight; i < gridHeight; i++) {
        if (sourceFlags[i * width + donor] == 1) {
            (*sourcesDown) += 1;
        }
    }
}
void set_targets_delayed(int reservoirHeight, int targetHeight, int gridHeight,
                         int width, int *sourceFlags, int *targetFlags,
                         int receiver, int *numTargets, int upSources,
                         int downSources) {

    int sourcesUp = upSources;
    int sourcesDown = downSources;

    for (int i = 0; i < reservoirHeight; i++) {
        if (sourceFlags[i * width + receiver] == 1) {
            sourcesUp += 1;
        }
    }

    for (int i = reservoirHeight + targetHeight; i < gridHeight; i++) {
        if (sourceFlags[i * width + receiver] == 1) {
            sourcesDown += 1;
        }
    }

    for (int i = 0; i < gridHeight; i++) {
        if ((i >= reservoirHeight) && (i < reservoirHeight + targetHeight)) {
            if (sourceFlags[i * width + receiver] == 0) {
                targetFlags[i] = 1;
                (*numTargets)++;
            } else {
                targetFlags[i] = 0;
            }
        } else {
            targetFlags[i] = 0;
        }
    }

    int i = reservoirHeight;

    while (sourcesUp > 0) {
        if (targetFlags[i] == 1) {
            targetFlags[i] = 0;
            sourcesUp -= 1;
            (*numTargets)--;
        }
        i++;
    }

    i = reservoirHeight + targetHeight;
    while (sourcesDown > 0) {
        if (targetFlags[i] == 1) {
            targetFlags[i] = 0;
            sourcesDown -= 1;
            (*numTargets)--;
        }
        i--;
    }
}

void set_targets(int reservoirHeight, int targetHeight, int gridHeight,
                 int width, int *sourceFlags, int *targetFlags, int receiver,
                 int *numTargets) {

    int sourcesUp = 0;
    int sourcesDown = 0;

    for (int i = 0; i < reservoirHeight; i++) {
        if (sourceFlags[i * width + receiver] == 1) {
            sourcesUp += 1;
        }
    }

    for (int i = reservoirHeight + targetHeight; i < gridHeight; i++) {
        if (sourceFlags[i * width + receiver] == 1) {
            sourcesDown += 1;
        }
    }

    for (int i = 0; i < gridHeight; i++) {
        if ((i >= reservoirHeight) && (i < reservoirHeight + targetHeight)) {
            if (sourceFlags[i * width + receiver] == 0) {
                targetFlags[i] = 1;
                (*numTargets)++;
            } else {
                targetFlags[i] = 0;
            }
        } else {
            targetFlags[i] = 0;
        }
    }

    int i = reservoirHeight;

    while (sourcesUp > 0) {
        if (targetFlags[i] == 1) {
            targetFlags[i] = 0;
            sourcesUp -= 1;
            (*numTargets)--;
        }
        i++;
    }

    i = reservoirHeight + targetHeight;
    while (sourcesDown > 0) {
        if (targetFlags[i] == 1) {
            targetFlags[i] = 0;
            sourcesDown -= 1;
            (*numTargets)--;
        }
        i--;
    }
}

void solve_receiver(int *ToMove, int *OutSources_cpu, int *OutTargets_cpu,
                    int reservoirHeight, int donor, int receiver,
                    int gridHeight, int width, int *sourceFlags,
                    int *currentIndex) {

    int targetIndex = reservoirHeight;
    append_up_sources(ToMove, OutSources_cpu, OutTargets_cpu, currentIndex,
                      reservoirHeight, width, donor, receiver, &targetIndex);
    for (int row = 0; row < gridHeight; row++) {
        if (sourceFlags[row * width + receiver] == 1) {
            OutSources_cpu[*currentIndex] = row * width + receiver;
            OutTargets_cpu[*currentIndex] = targetIndex * width + receiver;
            (*currentIndex)++;
            targetIndex++;
        }
    }
    append_down_sources(ToMove, OutSources_cpu, OutTargets_cpu, currentIndex,
                        gridHeight, reservoirHeight, width, donor, receiver,
                        &targetIndex);
}

void remove_atoms_from_donor(int *ToMove, int gridHeight, int width,
                             int *sourceFlags, int donor) {

    for (int i = 0; i < gridHeight; i++) {
        if (ToMove[i] == 1) {
            sourceFlags[i * width + donor] = 0;
        }
    }
}

// messes with matching
void redrec_cpu_v3_unbatched_moves(int gridHeight, int width,
                                   int reservoirHeight, int *sourceFlags,
                                   int *OutSources_cpu, int *OutTargets_cpu,
                                   int *outputMovesSource_cpu,
                                   int *outputMovesTarget_cpu,
                                   int *moves_counter) {

    int targetHeight = gridHeight - reservoirHeight * 2;
    int column_solved[width];
    int offset = 0;

    for (int i = 0; i < width; i++) {
        column_solved[i] = 0;
    }

    int surplus[width];
    int delayedMovingFlag[width];
    int delayedMovingStack[width * width];

    for (int i = 0; i < width; i++) {
        delayedMovingFlag[i] = -1;
    }

    for (int col = 0; col < width; col++) {
        int atomsCol = 0;
        for (int r = 0; r < gridHeight; r++) {
            if (sourceFlags[r * width + col] == 1) {
                atomsCol += 1;
            }
        }
        surplus[col] = atomsCol - targetHeight;
    }

    int dynamic_columns[width];

    for (int i = 0; i < width; i++) {
        dynamic_columns[i] = i;
    }

    int currentIndex = 0;

    for (int col = 0; col < width; col++) {
        if (surplus[col] == 0) {
            linear_solve(col, gridHeight, reservoirHeight, width, sourceFlags,
                         OutSources_cpu, OutTargets_cpu, &currentIndex,
                         reservoirHeight);
#ifdef moves_gen
            moves_generator_unbatched_column(
                width, OutSources_cpu, OutTargets_cpu, targetHeight,
                outputMovesSource_cpu, outputMovesTarget_cpu, moves_counter,
                &offset);
#endif
            column_solved[col] = 1;
            update_dynamic_columns(width, column_solved, dynamic_columns);
        }
    }

    while (deficit_exists(surplus, width)) {

        int donor, receiver;

        find_pair_to_shuffle(surplus, column_solved, dynamic_columns, &donor,
                             &receiver, width);
        int donor_surplus = surplus[donor];
        int receiver_deficit = abs(surplus[receiver]);

        if (delayedMovingFlag[receiver] == -1) {

            if (donor_surplus == receiver_deficit) {

                surplus[donor] = 0;
                surplus[receiver] = 0;
                column_solved[donor] = 1;
                column_solved[receiver] = 1;
                update_dynamic_columns(width, column_solved, dynamic_columns);

                int LocalsourceFlags[gridHeight];
                int targetFlags[gridHeight];
                int numSources = 0;
                int numTargets = 0;
                int ToMove[gridHeight];

                set_targets(reservoirHeight, targetHeight, gridHeight, width,
                            sourceFlags, targetFlags, receiver, &numTargets);
                set_sources(reservoirHeight, gridHeight, width, sourceFlags,
                            LocalsourceFlags, donor, &numSources);
                lin_exact_cpu_v2_flags(LocalsourceFlags, targetFlags,
                                       gridHeight, numSources, numTargets,
                                       ToMove);
                solve_receiver(ToMove, OutSources_cpu, OutTargets_cpu,
                               reservoirHeight, donor, receiver, gridHeight,
                               width, sourceFlags, &currentIndex);
#ifdef moves_gen
                moves_generator_unbatched_column(
                    width, OutSources_cpu, OutTargets_cpu, targetHeight,
                    outputMovesSource_cpu, outputMovesTarget_cpu, moves_counter,
                    &offset);
#endif
                remove_atoms_from_donor(ToMove, gridHeight, width, sourceFlags,
                                        donor);
                linear_solve(donor, gridHeight, reservoirHeight, width,
                             sourceFlags, OutSources_cpu, OutTargets_cpu,
                             &currentIndex, reservoirHeight);
#ifdef moves_gen
                moves_generator_unbatched_column(
                    width, OutSources_cpu, OutTargets_cpu, targetHeight,
                    outputMovesSource_cpu, outputMovesTarget_cpu, moves_counter,
                    &offset);
#endif
            }

            else if (donor_surplus > receiver_deficit) {

                surplus[donor] = donor_surplus - receiver_deficit;
                surplus[receiver] = 0;
                column_solved[receiver] = 1;
                update_dynamic_columns(width, column_solved, dynamic_columns);

                int LocalsourceFlags[gridHeight];
                int targetFlags[gridHeight];
                int numSources = 0;
                int numTargets = 0;
                int ToMove[gridHeight];

                set_targets(reservoirHeight, targetHeight, gridHeight, width,
                            sourceFlags, targetFlags, receiver, &numTargets);
                set_sources(reservoirHeight, gridHeight, width, sourceFlags,
                            LocalsourceFlags, donor, &numSources);
                lin_exact_cpu_v2_flags(LocalsourceFlags, targetFlags,
                                       gridHeight, numSources, numTargets,
                                       ToMove);
                solve_receiver(ToMove, OutSources_cpu, OutTargets_cpu,
                               reservoirHeight, donor, receiver, gridHeight,
                               width, sourceFlags, &currentIndex);
#ifdef moves_gen
                moves_generator_unbatched_column(
                    width, OutSources_cpu, OutTargets_cpu, targetHeight,
                    outputMovesSource_cpu, outputMovesTarget_cpu, moves_counter,
                    &offset);
#endif
                remove_atoms_from_donor(ToMove, gridHeight, width, sourceFlags,
                                        donor);

            } else {

                surplus[donor] = 0;
                surplus[receiver] = donor_surplus - receiver_deficit;
                column_solved[donor] = 1;
                delayedMovingFlag[receiver]++;
                delayedMovingStack[delayedMovingFlag[receiver] * width +
                                   receiver] = donor;
                update_dynamic_columns(width, column_solved, dynamic_columns);

                int target[gridHeight];
                int source[gridHeight];
                int ToMove[gridHeight];
                int targetIndex = reservoirHeight;
                int numSources = targetHeight + donor_surplus;
                for (int r = 0; r < gridHeight; r++) {
                    if ((r >= reservoirHeight) &&
                        (r < gridHeight - reservoirHeight)) {
                        target[r] = 1;
                    } else {
                        target[r] = 0;
                    }
                }
                for (int r = 0; r < gridHeight; r++) {
                    if (sourceFlags[r * width + donor] == 1) {
                        source[r] = 1;
                    } else {
                        source[r] = 0;
                    }
                }
                lin_exact_cpu_v2_flags(source, target, gridHeight, numSources,
                                       targetHeight, ToMove);
                for (int row = 0; row < gridHeight; row++) {
                    if (ToMove[row] == 1) {
                        OutSources_cpu[currentIndex] = row * width + donor;
                        OutTargets_cpu[currentIndex] =
                            targetIndex * width + donor;
                        currentIndex++;
                        targetIndex++;
                    }
                }
                remove_atoms_from_donor(ToMove, gridHeight, width, sourceFlags,
                                        donor);
#ifdef moves_gen
                moves_generator_unbatched_column(
                    width, OutSources_cpu, OutTargets_cpu, targetHeight,
                    outputMovesSource_cpu, outputMovesTarget_cpu, moves_counter,
                    &offset);
#endif
            }
        } else {
            if (donor_surplus == receiver_deficit) {
                surplus[donor] = 0;
                surplus[receiver] = 0;
                column_solved[donor] = 1;
                column_solved[receiver] = 1;
                update_dynamic_columns(width, column_solved, dynamic_columns);

                int upSources = 0;
                int downSources = 0;
                int targetFlags[gridHeight];
                int numTargets = 0;
                int LocalsourceFlags[gridHeight];
                int numSources = 0;
                int ToMove[gridHeight];
                int targetIndex = reservoirHeight;
                for (int i = delayedMovingFlag[receiver]; i >= 0; i--) {
                    int delayedDonor = delayedMovingStack[i * width + receiver];
                    count_up_sources(reservoirHeight, width, sourceFlags,
                                     delayedDonor, &upSources);
                    count_down_sources(gridHeight, targetHeight, width,
                                       sourceFlags, delayedDonor, &downSources);
                }
                set_targets_delayed(reservoirHeight, targetHeight, gridHeight,
                                    width, sourceFlags, targetFlags, receiver,
                                    &numTargets, upSources, downSources);
                set_sources(reservoirHeight, gridHeight, width, sourceFlags,
                            LocalsourceFlags, donor, &numSources);
                lin_exact_cpu_v2_flags(LocalsourceFlags, targetFlags,
                                       gridHeight, numSources, numTargets,
                                       ToMove);
                remove_atoms_from_donor(ToMove, gridHeight, width, sourceFlags,
                                        donor);
                append_up_sources(ToMove, OutSources_cpu, OutTargets_cpu,
                                  &currentIndex, reservoirHeight, width, donor,
                                  receiver, &targetIndex);
                for (int i = delayedMovingFlag[receiver]; i >= 0; i--) {
                    int delayedDonor = delayedMovingStack[i * width + receiver];
                    for (int row = 0; row < reservoirHeight; row++) {
                        if (sourceFlags[row * width + delayedDonor] == 1) {
                            OutSources_cpu[currentIndex] =
                                row * width + delayedDonor;
                            OutTargets_cpu[currentIndex] =
                                targetIndex * width + receiver;
                            currentIndex++;
                            targetIndex++;
                        }
                    }
                }
                for (int row = 0; row < gridHeight; row++) {
                    if (sourceFlags[row * width + receiver] == 1) {
                        OutSources_cpu[currentIndex] = row * width + receiver;
                        OutTargets_cpu[currentIndex] =
                            targetIndex * width + receiver;
                        currentIndex++;
                        targetIndex++;
                    }
                }
                for (int i = 0; i <= delayedMovingFlag[receiver]; i++) {
                    int delayedDonor = delayedMovingStack[i * width + receiver];
                    for (int row = gridHeight - reservoirHeight;
                         row < gridHeight; row++) {
                        if (sourceFlags[row * width + delayedDonor] == 1) {
                            OutSources_cpu[currentIndex] =
                                row * width + delayedDonor;
                            OutTargets_cpu[currentIndex] =
                                targetIndex * width + receiver;
                            currentIndex++;
                            targetIndex++;
                        }
                    }
                }
                append_down_sources(ToMove, OutSources_cpu, OutTargets_cpu,
                                    &currentIndex, gridHeight, reservoirHeight,
                                    width, donor, receiver, &targetIndex);
#ifdef moves_gen
                moves_generator_unbatched_column(
                    width, OutSources_cpu, OutTargets_cpu, targetHeight,
                    outputMovesSource_cpu, outputMovesTarget_cpu, moves_counter,
                    &offset);
#endif
                linear_solve(donor, gridHeight, reservoirHeight, width,
                             sourceFlags, OutSources_cpu, OutTargets_cpu,
                             &currentIndex, reservoirHeight);
#ifdef moves_gen
                moves_generator_unbatched_column(
                    width, OutSources_cpu, OutTargets_cpu, targetHeight,
                    outputMovesSource_cpu, outputMovesTarget_cpu, moves_counter,
                    &offset);
#endif
            }

            else if (donor_surplus > receiver_deficit) {
                surplus[donor] = donor_surplus - receiver_deficit;
                surplus[receiver] = 0;
                column_solved[receiver] = 1;
                update_dynamic_columns(width, column_solved, dynamic_columns);

                int upSources = 0;
                int downSources = 0;
                int targetFlags[gridHeight];
                int numTargets = 0;
                int LocalsourceFlags[gridHeight];
                int numSources = 0;
                int ToMove[gridHeight];
                int targetIndex = reservoirHeight;
                for (int i = delayedMovingFlag[receiver]; i >= 0; i--) {
                    int delayedDonor = delayedMovingStack[i * width + receiver];
                    count_up_sources(reservoirHeight, width, sourceFlags,
                                     delayedDonor, &upSources);
                    count_down_sources(gridHeight, targetHeight, width,
                                       sourceFlags, delayedDonor, &downSources);
                }
                set_targets_delayed(reservoirHeight, targetHeight, gridHeight,
                                    width, sourceFlags, targetFlags, receiver,
                                    &numTargets, upSources, downSources);
                set_sources(reservoirHeight, gridHeight, width, sourceFlags,
                            LocalsourceFlags, donor, &numSources);
                lin_exact_cpu_v2_flags(LocalsourceFlags, targetFlags,
                                       gridHeight, numSources, numTargets,
                                       ToMove);
                remove_atoms_from_donor(ToMove, gridHeight, width, sourceFlags,
                                        donor);
                append_up_sources(ToMove, OutSources_cpu, OutTargets_cpu,
                                  &currentIndex, reservoirHeight, width, donor,
                                  receiver, &targetIndex);
                for (int i = delayedMovingFlag[receiver]; i >= 0; i--) {
                    int delayedDonor = delayedMovingStack[i * width + receiver];
                    for (int row = 0; row < reservoirHeight; row++) {
                        if (sourceFlags[row * width + delayedDonor] == 1) {
                            OutSources_cpu[currentIndex] =
                                row * width + delayedDonor;
                            OutTargets_cpu[currentIndex] =
                                targetIndex * width + receiver;
                            currentIndex++;
                            targetIndex++;
                        }
                    }
                }
                for (int row = 0; row < gridHeight; row++) {
                    if (sourceFlags[row * width + receiver] == 1) {
                        OutSources_cpu[currentIndex] = row * width + receiver;
                        OutTargets_cpu[currentIndex] =
                            targetIndex * width + receiver;
                        currentIndex++;
                        targetIndex++;
                    }
                }
                for (int i = 0; i <= delayedMovingFlag[receiver]; i++) {
                    int delayedDonor = delayedMovingStack[i * width + receiver];
                    for (int row = gridHeight - reservoirHeight;
                         row < gridHeight; row++) {
                        if (sourceFlags[row * width + delayedDonor] == 1) {
                            OutSources_cpu[currentIndex] =
                                row * width + delayedDonor;
                            OutTargets_cpu[currentIndex] =
                                targetIndex * width + receiver;
                            currentIndex++;
                            targetIndex++;
                        }
                    }
                }
                append_down_sources(ToMove, OutSources_cpu, OutTargets_cpu,
                                    &currentIndex, gridHeight, reservoirHeight,
                                    width, donor, receiver, &targetIndex);
#ifdef moves_gen
                moves_generator_unbatched_column(
                    width, OutSources_cpu, OutTargets_cpu, targetHeight,
                    outputMovesSource_cpu, outputMovesTarget_cpu, moves_counter,
                    &offset);
#endif
            } else {
                surplus[donor] = 0;
                surplus[receiver] = donor_surplus - receiver_deficit;
                column_solved[donor] = 1;
                delayedMovingFlag[receiver]++;
                delayedMovingStack[delayedMovingFlag[receiver] * width +
                                   receiver] = donor;
                update_dynamic_columns(width, column_solved, dynamic_columns);

                int target[gridHeight];
                int source[gridHeight];
                int ToMove[gridHeight];
                int targetIndex = reservoirHeight;
                int numSources = targetHeight + donor_surplus;
                for (int r = 0; r < gridHeight; r++) {
                    if ((r >= reservoirHeight) &&
                        (r < gridHeight - reservoirHeight)) {
                        target[r] = 1;
                    } else {
                        target[r] = 0;
                    }
                }
                for (int r = 0; r < gridHeight; r++) {
                    if (sourceFlags[r * width + donor] == 1) {
                        source[r] = 1;
                    } else {
                        source[r] = 0;
                    }
                }
                lin_exact_cpu_v2_flags(source, target, gridHeight, numSources,
                                       targetHeight, ToMove);
                for (int row = 0; row < gridHeight; row++) {
                    if (ToMove[row] == 1) {
                        OutSources_cpu[currentIndex] = row * width + donor;
                        OutTargets_cpu[currentIndex] =
                            targetIndex * width + donor;
                        currentIndex++;
                        targetIndex++;
                    }
                }
                remove_atoms_from_donor(ToMove, gridHeight, width, sourceFlags,
                                        donor);
#ifdef moves_gen
                moves_generator_unbatched_column(
                    width, OutSources_cpu, OutTargets_cpu, targetHeight,
                    outputMovesSource_cpu, outputMovesTarget_cpu, moves_counter,
                    &offset);
#endif
            }
        }
    }

    int target[gridHeight];
    for (int r = 0; r < gridHeight; r++) {
        if ((r >= reservoirHeight) && (r < gridHeight - reservoirHeight)) {
            target[r] = 1;
        } else {
            target[r] = 0;
        }
    }

    for (int col = 0; col < width; col++) {
        if (!column_solved[col]) {

            int source[gridHeight];
            int ToMove[gridHeight];
            int numSources = targetHeight + surplus[col];
            int targetIndex = reservoirHeight;

            for (int r = 0; r < gridHeight; r++) {
                if (sourceFlags[r * width + col] == 1) {
                    source[r] = 1;
                } else {
                    source[r] = 0;
                }
            }
            lin_exact_cpu_v2_flags(source, target, gridHeight, numSources,
                                   targetHeight, ToMove);
            for (int row = 0; row < gridHeight; row++) {
                if (ToMove[row] == 1) {
                    OutSources_cpu[currentIndex] = row * width + col;
                    OutTargets_cpu[currentIndex] = targetIndex * width + col;
                    currentIndex++;
                    targetIndex++;
                }
            }
#ifdef moves_gen
            moves_generator_unbatched_column(
                width, OutSources_cpu, OutTargets_cpu, targetHeight,
                outputMovesSource_cpu, outputMovesTarget_cpu, moves_counter,
                &offset);
#endif
        }
    }
}

void moves_generator_unbatched_matching(int gridHeight, int width,
                                        int reservoirHeight, int *OutSources,
                                        int *OutTargets, int *outputMovesSource,
                                        int *outputMovesTarget,
                                        int *moves_counter) {
    int offset = 0;
    int targetHeight = gridHeight - reservoirHeight * 2;
    ;
    for (int i = 0; i < width; i++) {
        moves_generator_unbatched_column(
            width, OutSources, OutTargets, targetHeight, outputMovesSource,
            outputMovesTarget, moves_counter, &offset);
    }
}

void redrec_cpu_v3_matching(int gridHeight, int width, int reservoirHeight,
                            int *sourceFlags, int *OutSources_cpu,
                            int *OutTargets_cpu, int *outputMovesSource_cpu,
                            int *outputMovesTarget_cpu, int *moves_counter) {

    int targetHeight = gridHeight - reservoirHeight * 2;
    int column_solved[width];
    int offset = 0;

    for (int i = 0; i < width; i++) {
        column_solved[i] = 0;
    }

    int surplus[width];
    int delayedMovingFlag[width];
    int delayedMovingStack[width * width];

    for (int i = 0; i < width; i++) {
        delayedMovingFlag[i] = -1;
    }

    for (int col = 0; col < width; col++) {
        int atomsCol = 0;
        for (int r = 0; r < gridHeight; r++) {
            if (sourceFlags[r * width + col] == 1) {
                atomsCol += 1;
            }
        }
        surplus[col] = atomsCol - targetHeight;
    }

    int dynamic_columns[width];

    for (int i = 0; i < width; i++) {
        dynamic_columns[i] = i;
    }

    int currentIndex = 0;

    for (int col = 0; col < width; col++) {
        if (surplus[col] == 0) {
            linear_solve(col, gridHeight, reservoirHeight, width, sourceFlags,
                         OutSources_cpu, OutTargets_cpu, &currentIndex,
                         reservoirHeight);

            column_solved[col] = 1;
            update_dynamic_columns(width, column_solved, dynamic_columns);
        }
    }

    while (deficit_exists(surplus, width)) {

        int donor, receiver;

        find_pair_to_shuffle(surplus, column_solved, dynamic_columns, &donor,
                             &receiver, width);
        int donor_surplus = surplus[donor];
        int receiver_deficit = abs(surplus[receiver]);

        if (delayedMovingFlag[receiver] == -1) {

            if (donor_surplus == receiver_deficit) {

                surplus[donor] = 0;
                surplus[receiver] = 0;
                column_solved[donor] = 1;
                column_solved[receiver] = 1;
                update_dynamic_columns(width, column_solved, dynamic_columns);

                int LocalsourceFlags[gridHeight];
                int targetFlags[gridHeight];
                int numSources = 0;
                int numTargets = 0;
                int ToMove[gridHeight];

                set_targets(reservoirHeight, targetHeight, gridHeight, width,
                            sourceFlags, targetFlags, receiver, &numTargets);
                set_sources(reservoirHeight, gridHeight, width, sourceFlags,
                            LocalsourceFlags, donor, &numSources);
                lin_exact_cpu_v2_flags(LocalsourceFlags, targetFlags,
                                       gridHeight, numSources, numTargets,
                                       ToMove);
                solve_receiver(ToMove, OutSources_cpu, OutTargets_cpu,
                               reservoirHeight, donor, receiver, gridHeight,
                               width, sourceFlags, &currentIndex);

                remove_atoms_from_donor(ToMove, gridHeight, width, sourceFlags,
                                        donor);
                linear_solve(donor, gridHeight, reservoirHeight, width,
                             sourceFlags, OutSources_cpu, OutTargets_cpu,
                             &currentIndex, reservoirHeight);

            }

            else if (donor_surplus > receiver_deficit) {

                surplus[donor] = donor_surplus - receiver_deficit;
                surplus[receiver] = 0;
                column_solved[receiver] = 1;
                update_dynamic_columns(width, column_solved, dynamic_columns);

                int LocalsourceFlags[gridHeight];
                int targetFlags[gridHeight];
                int numSources = 0;
                int numTargets = 0;
                int ToMove[gridHeight];

                set_targets(reservoirHeight, targetHeight, gridHeight, width,
                            sourceFlags, targetFlags, receiver, &numTargets);
                set_sources(reservoirHeight, gridHeight, width, sourceFlags,
                            LocalsourceFlags, donor, &numSources);
                lin_exact_cpu_v2_flags(LocalsourceFlags, targetFlags,
                                       gridHeight, numSources, numTargets,
                                       ToMove);
                solve_receiver(ToMove, OutSources_cpu, OutTargets_cpu,
                               reservoirHeight, donor, receiver, gridHeight,
                               width, sourceFlags, &currentIndex);

                remove_atoms_from_donor(ToMove, gridHeight, width, sourceFlags,
                                        donor);

            } else {

                surplus[donor] = 0;
                surplus[receiver] = donor_surplus - receiver_deficit;
                column_solved[donor] = 1;
                delayedMovingFlag[receiver]++;
                delayedMovingStack[delayedMovingFlag[receiver] * width +
                                   receiver] = donor;
                update_dynamic_columns(width, column_solved, dynamic_columns);

                int target[gridHeight];
                int source[gridHeight];
                int ToMove[gridHeight];
                int targetIndex = reservoirHeight;
                int numSources = targetHeight + donor_surplus;
                for (int r = 0; r < gridHeight; r++) {
                    if ((r >= reservoirHeight) &&
                        (r < gridHeight - reservoirHeight)) {
                        target[r] = 1;
                    } else {
                        target[r] = 0;
                    }
                }
                for (int r = 0; r < gridHeight; r++) {
                    if (sourceFlags[r * width + donor] == 1) {
                        source[r] = 1;
                    } else {
                        source[r] = 0;
                    }
                }
                lin_exact_cpu_v2_flags(source, target, gridHeight, numSources,
                                       targetHeight, ToMove);
                for (int row = 0; row < gridHeight; row++) {
                    if (ToMove[row] == 1) {
                        OutSources_cpu[currentIndex] = row * width + donor;
                        OutTargets_cpu[currentIndex] =
                            targetIndex * width + donor;
                        currentIndex++;
                        targetIndex++;
                    }
                }
                remove_atoms_from_donor(ToMove, gridHeight, width, sourceFlags,
                                        donor);
            }
        } else {
            if (donor_surplus == receiver_deficit) {
                surplus[donor] = 0;
                surplus[receiver] = 0;
                column_solved[donor] = 1;
                column_solved[receiver] = 1;
                update_dynamic_columns(width, column_solved, dynamic_columns);

                int upSources = 0;
                int downSources = 0;
                int targetFlags[gridHeight];
                int numTargets = 0;
                int LocalsourceFlags[gridHeight];
                int numSources = 0;
                int ToMove[gridHeight];
                int targetIndex = reservoirHeight;
                for (int i = delayedMovingFlag[receiver]; i >= 0; i--) {
                    int delayedDonor = delayedMovingStack[i * width + receiver];
                    count_up_sources(reservoirHeight, width, sourceFlags,
                                     delayedDonor, &upSources);
                    count_down_sources(gridHeight, targetHeight, width,
                                       sourceFlags, delayedDonor, &downSources);
                }
                set_targets_delayed(reservoirHeight, targetHeight, gridHeight,
                                    width, sourceFlags, targetFlags, receiver,
                                    &numTargets, upSources, downSources);
                set_sources(reservoirHeight, gridHeight, width, sourceFlags,
                            LocalsourceFlags, donor, &numSources);
                lin_exact_cpu_v2_flags(LocalsourceFlags, targetFlags,
                                       gridHeight, numSources, numTargets,
                                       ToMove);
                remove_atoms_from_donor(ToMove, gridHeight, width, sourceFlags,
                                        donor);
                append_up_sources(ToMove, OutSources_cpu, OutTargets_cpu,
                                  &currentIndex, reservoirHeight, width, donor,
                                  receiver, &targetIndex);
                for (int i = delayedMovingFlag[receiver]; i >= 0; i--) {
                    int delayedDonor = delayedMovingStack[i * width + receiver];
                    for (int row = 0; row < reservoirHeight; row++) {
                        if (sourceFlags[row * width + delayedDonor] == 1) {
                            OutSources_cpu[currentIndex] =
                                row * width + delayedDonor;
                            OutTargets_cpu[currentIndex] =
                                targetIndex * width + receiver;
                            currentIndex++;
                            targetIndex++;
                        }
                    }
                }
                for (int row = 0; row < gridHeight; row++) {
                    if (sourceFlags[row * width + receiver] == 1) {
                        OutSources_cpu[currentIndex] = row * width + receiver;
                        OutTargets_cpu[currentIndex] =
                            targetIndex * width + receiver;
                        currentIndex++;
                        targetIndex++;
                    }
                }
                for (int i = 0; i <= delayedMovingFlag[receiver]; i++) {
                    int delayedDonor = delayedMovingStack[i * width + receiver];
                    for (int row = gridHeight - reservoirHeight;
                         row < gridHeight; row++) {
                        if (sourceFlags[row * width + delayedDonor] == 1) {
                            OutSources_cpu[currentIndex] =
                                row * width + delayedDonor;
                            OutTargets_cpu[currentIndex] =
                                targetIndex * width + receiver;
                            currentIndex++;
                            targetIndex++;
                        }
                    }
                }
                append_down_sources(ToMove, OutSources_cpu, OutTargets_cpu,
                                    &currentIndex, gridHeight, reservoirHeight,
                                    width, donor, receiver, &targetIndex);

                linear_solve(donor, gridHeight, reservoirHeight, width,
                             sourceFlags, OutSources_cpu, OutTargets_cpu,
                             &currentIndex, reservoirHeight);
            }

            else if (donor_surplus > receiver_deficit) {
                surplus[donor] = donor_surplus - receiver_deficit;
                surplus[receiver] = 0;
                column_solved[receiver] = 1;
                update_dynamic_columns(width, column_solved, dynamic_columns);

                int upSources = 0;
                int downSources = 0;
                int targetFlags[gridHeight];
                int numTargets = 0;
                int LocalsourceFlags[gridHeight];
                int numSources = 0;
                int ToMove[gridHeight];
                int targetIndex = reservoirHeight;
                for (int i = delayedMovingFlag[receiver]; i >= 0; i--) {
                    int delayedDonor = delayedMovingStack[i * width + receiver];
                    count_up_sources(reservoirHeight, width, sourceFlags,
                                     delayedDonor, &upSources);
                    count_down_sources(gridHeight, targetHeight, width,
                                       sourceFlags, delayedDonor, &downSources);
                }
                set_targets_delayed(reservoirHeight, targetHeight, gridHeight,
                                    width, sourceFlags, targetFlags, receiver,
                                    &numTargets, upSources, downSources);
                set_sources(reservoirHeight, gridHeight, width, sourceFlags,
                            LocalsourceFlags, donor, &numSources);
                lin_exact_cpu_v2_flags(LocalsourceFlags, targetFlags,
                                       gridHeight, numSources, numTargets,
                                       ToMove);
                remove_atoms_from_donor(ToMove, gridHeight, width, sourceFlags,
                                        donor);
                append_up_sources(ToMove, OutSources_cpu, OutTargets_cpu,
                                  &currentIndex, reservoirHeight, width, donor,
                                  receiver, &targetIndex);
                for (int i = delayedMovingFlag[receiver]; i >= 0; i--) {
                    int delayedDonor = delayedMovingStack[i * width + receiver];
                    for (int row = 0; row < reservoirHeight; row++) {
                        if (sourceFlags[row * width + delayedDonor] == 1) {
                            OutSources_cpu[currentIndex] =
                                row * width + delayedDonor;
                            OutTargets_cpu[currentIndex] =
                                targetIndex * width + receiver;
                            currentIndex++;
                            targetIndex++;
                        }
                    }
                }
                for (int row = 0; row < gridHeight; row++) {
                    if (sourceFlags[row * width + receiver] == 1) {
                        OutSources_cpu[currentIndex] = row * width + receiver;
                        OutTargets_cpu[currentIndex] =
                            targetIndex * width + receiver;
                        currentIndex++;
                        targetIndex++;
                    }
                }
                for (int i = 0; i <= delayedMovingFlag[receiver]; i++) {
                    int delayedDonor = delayedMovingStack[i * width + receiver];
                    for (int row = gridHeight - reservoirHeight;
                         row < gridHeight; row++) {
                        if (sourceFlags[row * width + delayedDonor] == 1) {
                            OutSources_cpu[currentIndex] =
                                row * width + delayedDonor;
                            OutTargets_cpu[currentIndex] =
                                targetIndex * width + receiver;
                            currentIndex++;
                            targetIndex++;
                        }
                    }
                }
                append_down_sources(ToMove, OutSources_cpu, OutTargets_cpu,
                                    &currentIndex, gridHeight, reservoirHeight,
                                    width, donor, receiver, &targetIndex);
            } else {
                surplus[donor] = 0;
                surplus[receiver] = donor_surplus - receiver_deficit;
                column_solved[donor] = 1;
                delayedMovingFlag[receiver]++;
                delayedMovingStack[delayedMovingFlag[receiver] * width +
                                   receiver] = donor;
                update_dynamic_columns(width, column_solved, dynamic_columns);

                int target[gridHeight];
                int source[gridHeight];
                int ToMove[gridHeight];
                int targetIndex = reservoirHeight;
                int numSources = targetHeight + donor_surplus;
                for (int r = 0; r < gridHeight; r++) {
                    if ((r >= reservoirHeight) &&
                        (r < gridHeight - reservoirHeight)) {
                        target[r] = 1;
                    } else {
                        target[r] = 0;
                    }
                }
                for (int r = 0; r < gridHeight; r++) {
                    if (sourceFlags[r * width + donor] == 1) {
                        source[r] = 1;
                    } else {
                        source[r] = 0;
                    }
                }
                lin_exact_cpu_v2_flags(source, target, gridHeight, numSources,
                                       targetHeight, ToMove);
                for (int row = 0; row < gridHeight; row++) {
                    if (ToMove[row] == 1) {
                        OutSources_cpu[currentIndex] = row * width + donor;
                        OutTargets_cpu[currentIndex] =
                            targetIndex * width + donor;
                        currentIndex++;
                        targetIndex++;
                    }
                }
                remove_atoms_from_donor(ToMove, gridHeight, width, sourceFlags,
                                        donor);
            }
        }
    }

    int target[gridHeight];
    for (int r = 0; r < gridHeight; r++) {
        if ((r >= reservoirHeight) && (r < gridHeight - reservoirHeight)) {
            target[r] = 1;
        } else {
            target[r] = 0;
        }
    }

    for (int col = 0; col < width; col++) {
        if (!column_solved[col]) {

            int source[gridHeight];
            int ToMove[gridHeight];
            int numSources = targetHeight + surplus[col];
            int targetIndex = reservoirHeight;

            for (int r = 0; r < gridHeight; r++) {
                if (sourceFlags[r * width + col] == 1) {
                    source[r] = 1;
                } else {
                    source[r] = 0;
                }
            }
            lin_exact_cpu_v2_flags(source, target, gridHeight, numSources,
                                   targetHeight, ToMove);
            for (int row = 0; row < gridHeight; row++) {
                if (ToMove[row] == 1) {
                    OutSources_cpu[currentIndex] = row * width + col;
                    OutTargets_cpu[currentIndex] = targetIndex * width + col;
                    currentIndex++;
                    targetIndex++;
                }
            }
        }
    }
}
