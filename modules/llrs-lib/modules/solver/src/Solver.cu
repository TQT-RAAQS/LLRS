#include "Solver.h"

/**
 * @brief Sums the elements of a vector to find the number of the atoms
 *
 * @param atom_config The vector containing the configuration of the atoms
 * @return int Number of atoms in the configuration
 */
int count_num_atoms(std::vector<int32_t> atom_config) {
    return std::accumulate(atom_config.begin(), atom_config.end(),
                           decltype(atom_config)::value_type(0));
}

/**
 * @brief Check if all target atoms exists in current trap
 * Loop through the target and current configuration and check if all the target
 * atoms exist in the current trap
 * @param current
 * @param target
 * @return true
 * @return false
 */
bool target_met(std::vector<int32_t> current, std::vector<int32_t> target) {
    for (size_t trap = 0; trap < target.size() && trap < current.size();
         ++trap) {
        if (target[trap] && !current[trap]) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Get the algo enum object from the string name
 *
 * @param name
 * @return Algo
 * @throw std::invalid_argument if the name is not a valid algorithm name
 */

Reconfig::Algo get_algo_enum(std::string name) {
    if (name == "LINEAR-EXACT-1D") {
        return Reconfig::LINEAR_EXACT_1D;
    } else if (name == "LINEAR-EXACT-V2-1D") {
        return Reconfig::LINEAR_EXACT_V2_1D;
    } else if (name == "REDREC-V2-2D") {
        return Reconfig::REDREC_CPU_V2_2D;
    } else if (name == "REDREC-CPU-V3-2D") {
        return Reconfig::REDREC_CPU_V3_2D;
    } else if (name == "ARO-2D") {
        return Reconfig::ARO_CPU_2D;
    } else if (name == "REDREC-GPU-V3-2D") {
        return Reconfig::REDREC_GPU_V3_2D;
    } else if (name == "BIRD-CPU-2D") {
        return Reconfig::BIRD_CPU_2D;
    } else {
        throw std::invalid_argument(
            "Algorithm not supported"); /// indicates that some other algorithm
                                        /// has been provided which is not
                                        /// supported
    }
}

/// Solver Class

/**
 * @brief Construct a new Solver object
 *
 * @param Nt_x
 * @param Nt_y
 */
Reconfig::Solver::Solver(int Nt_x, int Nt_y, Util::Collector *p_collector)
    : Nt_x{Nt_x}, Nt_y{Nt_y}, num_atoms_initial{0}, num_atoms_target{0},
      matching_src(Nt_x * Nt_y, 0), matching_dst(Nt_x * Nt_y, 0),
      src(Nt_x * Nt_x * Nt_y * Nt_y, 0), dst(Nt_x * Nt_x * Nt_y * Nt_y, 0),
      blk(Nt_x * Nt_x * Nt_y * Nt_y, 0),
      batch_ptrs(Nt_x * Nt_x * Nt_y * Nt_y, 0),
      path_system(Nt_x * Nt_x * Nt_y * Nt_y, 0),
      path_length(Nt_x * Nt_y, 0), initial{}, p_collector{p_collector} {}

/**
 *   @brief Start the solver with a selected algorithm, initial and target
 * configurations.
 *   @param algo_select The algorithm to use for solving.
 *   @param initial The initial configuration of trap states.
 *   @param target The target configuration of trap states.
 *   @return LLRS_OK if the solver was started successfully, LLRS_ERR otherwise.
 *   @throws std::invalid_argument if the trap dimensions are invalid for the 1D
 * solver or if the size of the initial and target configurations do not match.
 */
bool Reconfig::Solver::start_solver(Algo algo_select,
                                    std::vector<int32_t> initial,
                                    std::vector<int32_t> target, int trial_num,
                                    int rep_num, int cycle_num) {

    int sol_length, num_batches = 0;

    /// ensure that initial and target configurations are of the same size and
    /// that the initial configuration is  equivalent to the full array of traps
    /// specified by Nt_x and Nt_y
    if (Nt_x * Nt_y != initial.size() || initial.size() != target.size()) {
        throw std::invalid_argument(
            "Invalid size of initial and target configuration.");
    }

    /// count the number of atoms in the initial configuration
    num_atoms_initial = count_num_atoms(initial);
    num_atoms_target = count_num_atoms(target);
    /// Copy the initial vector given
    this->initial = initial;

    /// return an error if the target configuration requires more atoms than we
    /// initially have
    if (num_atoms_target > num_atoms_initial) {
        return LLRS_ERR;
    }

    int reservoir_height = (Nt_y - num_atoms_target / Nt_x) / 2;

    /// Depending on the algorithm indicated in the function call, apply the
    /// desired algorithm for the problem
    switch (algo_select) {
    case LINEAR_EXACT_1D:

        lin_exact_block_batched_cpu(&initial[0], &target[0], initial.size(),
                                    num_atoms_initial, num_atoms_target,
                                    &src[0], &dst[0], &blk[0], &batch_ptrs[0],
                                    &num_batches, &sol_length);

        break;

    case LINEAR_EXACT_V2_1D:
        /// Matching
#ifdef LOGGING_RUNTIME
        p_collector->start_timer("III-Matching", trial_num, rep_num, cycle_num);
#endif
        lin_exact_cpu_v2_generate_matching // if supported algorithm is provided
                                           // then perform matching
            (&initial[0], &target[0], initial.size(), num_atoms_initial,
             num_atoms_target, &matching_src[0], &matching_dst[0]);
#ifdef LOGGING_RUNTIME
        p_collector->end_timer("III-Matching", trial_num, rep_num, cycle_num);
        p_collector->start_timer("III-Batching", trial_num, rep_num, cycle_num);
#endif
        /// Batching
        lin_exact_1d_cpu_v2_block_output_generator(
            &matching_src[0], &matching_dst[0], num_atoms_target, &src[0],
            &dst[0], &blk[0], &batch_ptrs[0], &num_batches, &sol_length);
        break;
#ifdef LOGGING_RUNTIME
        p_collector->end_timer("III-Batching", trial_num, rep_num, cycle_num);
#endif
    case REDREC_CPU_V2_2D:
#ifdef LOGGING_RUNTIME
        p_collector->start_timer("III-Matching", trial_num, rep_num, cycle_num);
#endif
        redrec_v2 /// gets batching time internally
            (&initial[0], num_atoms_initial, Nt_x, Nt_y, reservoir_height,
             &src[0], &dst[0], &batch_ptrs[0], &num_batches, &sol_length);
#ifdef LOGGING_RUNTIME
        p_collector->end_timer("III-Matching", trial_num, rep_num, cycle_num);
#endif
        break;
    case ARO_CPU_2D:
#ifdef LOGGING_RUNTIME
        p_collector->start_timer("III-Matching", trial_num, rep_num, cycle_num);
#endif
        aro_serial_cpu(&initial[0], &target[0], Nt_y, Nt_x, num_atoms_initial,
                       num_atoms_target, 1, 0,
                       1, /// do rerouting, don't do atom_isolation, do ordering
                       &src[0], &dst[0], &sol_length, &path_system[0],
                       &path_length[0]);
#ifdef LOGGING_RUNTIME
        p_collector->end_timer("III-Matching", trial_num, rep_num, cycle_num);
#endif
        break;
    case REDREC_CPU_V3_2D:
#ifdef LOGGING_RUNTIME
        p_collector->start_timer("III-Matching", trial_num, rep_num, cycle_num);
#endif
        redrec_cpu(Nt_y, Nt_x, reservoir_height, &initial[0], &matching_src[0],
                   &matching_dst[0], &src[0], &dst[0], &sol_length,
                   &path_system[0], &path_length[0]);
#ifdef LOGGING_RUNTIME
        p_collector->end_timer("III-Matching", trial_num, rep_num, cycle_num);
#endif
        break;
    case REDREC_GPU_V3_2D:
#ifdef LOGGING_RUNTIME
        p_collector->start_timer("III-Matching", trial_num, rep_num, cycle_num);
#endif
        redrec_gpu(Nt_y, Nt_x, reservoir_height, &initial[0], &matching_src[0],
                   &matching_dst[0], &src[0], &dst[0], &sol_length,
                   &path_system[0], &path_length[0]);
#ifdef LOGGING_RUNTIME
        p_collector->end_timer("III-Matching", trial_num, rep_num, cycle_num);
#endif
        break;
    case BIRD_CPU_2D:
#ifdef LOGGING_RUNTIME
        p_collector->start_timer("III-Matching", trial_num, rep_num, cycle_num);
#endif
        bird_cpu(Nt_y, Nt_x, reservoir_height, &initial[0], &matching_src[0],
                 &matching_dst[0], &src[0], &dst[0], &sol_length,
                 &path_system[0], &path_length[0]);
#ifdef LOGGING_RUNTIME
        p_collector->end_timer("III-Matching", trial_num, rep_num, cycle_num);
#endif
        break;
    default:
        throw std::invalid_argument("Algorithm not supported");
    }

    /// resize the member variables to be the length of the solution moveset
    /// given by the algorithm
    this->src.resize(sol_length);
    this->dst.resize(sol_length);
    this->blk.resize(sol_length);
    this->batch_ptrs.resize(num_batches);

    return LLRS_OK;
}

/**
 *   @brief Generate list of moves for reconfiguration algorithm
 *   @param algo_select An enum value specifying which reconfiguration algorithm
 * to use
 *   @return A vector of Move tuples representing the primary waveform keying
 * components
 */
std::vector<Reconfig::Move> Reconfig::Solver::gen_moves_list(Algo algo_select,
                                                             int trial_num,
                                                             int rep_num,
                                                             int cycle_num) {

    std::vector<Reconfig::Move> ret;
    switch (algo_select) {

    case LINEAR_EXACT_V2_1D:
    case LINEAR_EXACT_1D: {
        size_t paddings_required = src.size() < WF_PER_SEG
                                       ? 2 * WF_PER_SEG - src.size()
                                       : WF_PER_SEG - src.size() % WF_PER_SEG;
        ret.reserve(src.size() + paddings_required);
        for (size_t k = 0; k < src.size(); ++k) {
            size_t src_val = src[k];
            size_t dst_val = dst[k];
            size_t blk_val = blk[k];
            if (blk_val == 0)
                break;
            size_t wf_index = (src_val < dst_val)
                                  ? src_val
                                  : dst_val; /// change along the x-axis
            Synthesis::WfMoveType wf_type = (src_val < dst_val)
                                                ? Synthesis::FORWARD_1D
                                                : Synthesis::BACKWARD_1D;
            ret.emplace_back(wf_type, wf_index, 0, blk_val, Nt_x * Nt_y);
        }
        for (size_t k = 0; k < paddings_required; ++k) {
            ret.emplace_back(Synthesis::IDLE_1D, 0, 0, Nt_x, Nt_x * Nt_y);
        }
        return ret;
    }
    case REDREC_CPU_V2_2D: {
#ifdef LOGGING_RUNTIME
        p_collector->start_timer("III-Batching", trial_num, rep_num, cycle_num);
#endif
        for (auto itr = batch_ptrs.begin(); itr != batch_ptrs.end();
             itr++) // iterate through the batchings
        {
            auto tail = std::next(itr);
            size_t end = (tail != batch_ptrs.end()) ? *tail : src.size();
            size_t k = *itr;
            while (k < end) {
                bool is_red_not_rec = abs(src[k] - dst[k]) ==
                                      1; // determines if it is a redistribution
                                         // or recombination move
                bool is_reversed = src[k] > dst[k];

                /// classify the move as an enum depending on the boolean
                /// parameters above
                Synthesis::WfMoveType page =
                    (is_reversed) ? ((is_red_not_rec) ? Synthesis::LEFT_2D
                                                      : Synthesis::DOWN_2D)
                                  : ((is_red_not_rec) ? Synthesis::RIGHT_2D
                                                      : Synthesis::UP_2D);

                /// offset and index describe the location of a trap in the
                /// array, either _dst[k] or _src[k] depending on if the move is
                /// reversed or not, since we store the "smaller" of the two
                /// traps as the identifying index of the move
                int offset = (is_reversed) ? dst[k] % Nt_x : src[k] % Nt_x;
                int index = (is_reversed) ? dst[k] / Nt_x : src[k] / Nt_x;
                int blk_size = 1;

                if (is_red_not_rec) {
                    ret.emplace_back( // redistribution tuple added to back of
                                      // ret
                        Synthesis::EXTRACT_2D, src[k] / Nt_x, src[k] % Nt_x,
                        blk_size, 0 // extract the first _src[k] % _Nt_x traps
                                    // of the row containing _src[k]
                    );
                } else { // recombination extraction
                    if (k == *itr) {
                        if (!ret.empty()) {
                            if (std::get<1>(ret.back()) != offset ||
                                std::get<3>(ret.back()) < Nt_y) {
                                ret.emplace_back(Synthesis::EXTRACT_2D, 0,
                                                 offset, Nt_y, 0);
                            } else {
                                ret.pop_back();
                            }
                        } else {
                            ret.emplace_back(Synthesis::EXTRACT_2D, 0, offset,
                                             Nt_y, 0);
                        }
                    }
                }

                while (
                    ++k < end &&
                    (dst[k - 1] == src[k] ^
                     dst[k] == src[k - 1])) { // block adjacent moves together
                    blk_size++;
                }
                ret.emplace_back(page, index, offset, blk_size,
                                 Nt_y); // make a tuple of the move block

                // implantation move
                if (is_red_not_rec) {
                    ret.emplace_back(Synthesis::IMPLANT_2D,
                                     dst[k - blk_size] / Nt_x,
                                     dst[k - blk_size] % Nt_x, blk_size, 0);
                } else {
                    if (k >= end) {
                        ret.emplace_back(Synthesis::IMPLANT_2D, 0, offset, Nt_y,
                                         0);
                    }
                }
            }
        }
#ifdef LOGGING_RUNTIME
        p_collector->end_timer("III-Batching", trial_num, rep_num, cycle_num);
#endif
        break;
    }

    case BIRD_CPU_2D:
    case REDREC_GPU_V3_2D:
    case REDREC_CPU_V3_2D: {
#ifdef LOGGING_RUNTIME
        p_collector->start_timer("III-Batching", trial_num, rep_num, cycle_num);
#endif
        std::vector<int> single_atom_start, single_atom_end, single_atom_dir;
        int i = 0;
        while (i < src.size()) {
            single_atom_start.push_back(src[i]);
            single_atom_dir.push_back(dst[i] - src[i]);
            while (++i < src.size()) {
                bool batchable = dst[i - 1] == src[i];
                bool horizontal = abs(src[i - 1] - dst[i - 1]) == 1;
                bool same_dir = src[i - 1] - dst[i - 1] == src[i] - dst[i];
                if (!batchable || horizontal || !same_dir)
                    break;
            }
            single_atom_end.push_back(dst[i - 1]);
        }

        i = 0;
        while (i < single_atom_start.size()) {
            int min_trap = std::min(single_atom_start[i], single_atom_end[i]);
            int max_trap = std::max(single_atom_start[i], single_atom_end[i]);
            bool horizontal = abs(single_atom_dir[i]) == 1;

            int j = i + 1;
            if (!horizontal) {
                while (j < single_atom_start.size()) {
                    bool same_dir =
                        single_atom_dir[j] == single_atom_dir[j - 1];
                    bool same_offset = single_atom_start[j] % Nt_x ==
                                       single_atom_start[j - 1] % Nt_x;
                    if (!same_dir || !same_offset)
                        break;

                    min_trap = std::min(min_trap, std::min(single_atom_start[j],
                                                           single_atom_end[j]));
                    max_trap = std::max(max_trap, std::max(single_atom_start[j],
                                                           single_atom_end[j]));
                    ++j;
                }
            }
            int offset = min_trap % Nt_x;
            int index = min_trap / Nt_x;
            Synthesis::WfMoveType displacement_page;

            if (horizontal) {
                displacement_page = (single_atom_dir[i] == 1)
                                        ? Synthesis::RIGHT_2D
                                        : Synthesis::LEFT_2D;

                ret.emplace_back(Synthesis::EXTRACT_2D, index,
                                 single_atom_start[i] % Nt_x, 1, 0);
                ret.emplace_back(displacement_page, index, offset, 1, 0);
                ret.emplace_back(Synthesis::IMPLANT_2D, index,
                                 single_atom_end[i] % Nt_x, 1, 0);
                ++i;
                continue;
            }

            bool up = (single_atom_dir[i] > 0);
            displacement_page = up ? Synthesis::UP_2D : Synthesis::DOWN_2D;
            int extraction_extent = up ? max_trap / Nt_x + 1 : Nt_y - index;
            ret.emplace_back(Synthesis::EXTRACT_2D, up ? 0 : index, offset,
                             extraction_extent,
                             0); // block_size == extraction_extent here
            /// it may seem like this is turning the time complexity of the
            /// translation into quadratic however, it's really just reordering
            /// the moves returned by the algorithm it's quadratic w.r.t. size
            /// of single_atom_* but still linear w.r.t. size of _src & _dst
            while (true) {
                min_trap = Nt_x * Nt_y;
                max_trap = -1;
                for (int k = i; k < j; ++k) {
                    if (single_atom_start[k] != single_atom_end[k]) {
                        min_trap = std::min(min_trap,
                                            std::min(single_atom_start[k],
                                                     single_atom_start[k] +
                                                         single_atom_dir[i]));
                        max_trap = std::max(max_trap,
                                            std::max(single_atom_start[k],
                                                     single_atom_start[k] +
                                                         single_atom_dir[i]));
                        single_atom_start[k] += single_atom_dir[i];
                    }
                }
                if (max_trap == -1)
                    break;
                int blk_size = (max_trap - min_trap) / Nt_x;
                ret.emplace_back(displacement_page, min_trap / Nt_x, offset,
                                 blk_size, extraction_extent);
            }
            ret.emplace_back(Synthesis::IMPLANT_2D, up ? 0 : index, offset,
                             extraction_extent,
                             0); // block_size == extraction_extent here
            i = j;
        }
#ifdef LOGGING_RUNTIME
        p_collector->end_timer("III-Batching", trial_num, rep_num, cycle_num);
#endif
        break;
    }
    case ARO_CPU_2D: {
#ifdef DEBUG_BATCHING_ARO
        // Prining ungrouped moves
        std::cout << "UNGROUPED MOVES" << std::endl;
        for (int idx = 0; idx < _src.size(); idx++) {
            std::cout << "SRC: " << _src[idx] << " DEST: " << _dst[idx]
                      << std::endl;
        }
#endif
#ifdef LOGGING_RUNTIME
        p_collector->start_timer("III-Batching", trial_num, rep_num, cycle_num);
#endif

        std::vector<int> single_atom_start, single_atom_end, single_atom_dir;
        int i = 0;
        while (i < src.size()) {
            single_atom_start.push_back(src[i]);
            single_atom_dir.push_back(dst[i] - src[i]);
            while (++i < src.size()) {
                bool batchable = dst[i - 1] == src[i];
                bool horizontal = abs(src[i - 1] - dst[i - 1]) == 1;
                bool same_dir = src[i - 1] - dst[i - 1] == src[i] - dst[i];
                if (!batchable || horizontal || !same_dir)
                    break;
            }
            single_atom_end.push_back(dst[i - 1]);
        }

#ifdef DEBUG_BATCHING_ARO
        // Prining grouped moves
        std::cout << "GROUPED MOVES" << std::endl;
        for (int idx = 0; idx < single_atom_start.size(); idx++) {
            std::cout << "SRC: " << single_atom_start[idx]
                      << " DEST: " << single_atom_end[idx] << std::endl;
        }
#endif

        /// creating binary array to track locations of atoms between moves
        std::vector<int> atoms{};
        atoms.reserve(Nt_x * Nt_y);
        for (int idx = 0; idx < initial.size(); idx++) {
            atoms[idx] = initial[idx];
        }

        i = 0;
        while (i < single_atom_start.size()) {
            int min_trap = std::min(single_atom_start[i], single_atom_end[i]);
            int max_trap = std::max(single_atom_start[i], single_atom_end[i]);
            bool horizontal = abs(single_atom_dir[i]) == 1;

            int j = i + 1;
            if (!horizontal) {
                while (j < single_atom_start.size()) {
                    bool same_dir =
                        (single_atom_dir[j] == single_atom_dir[j - 1]);
                    bool same_dist =
                        ((single_atom_end[j] - single_atom_start[j]) ==
                         (single_atom_end[j - 1] - single_atom_start[j - 1]));
                    bool same_offset = ((single_atom_start[j] % Nt_x) ==
                                        (single_atom_start[j - 1] % Nt_x));
                    if (!same_dir || !same_offset)
                        break;

                    int new_min =
                        std::min(min_trap, std::min(single_atom_start[j],
                                                    single_atom_end[j]));
                    int new_max =
                        std::max(max_trap, std::max(single_atom_start[j],
                                                    single_atom_end[j]));
                    bool moving_extra_atom = false;
                    if (single_atom_dir[i] > 0) { // dest > src -> downward
                        if (new_min != min_trap || new_max != max_trap) {
                            int lower_range = 0;
                            int upper_range = 0;
                            if (new_min == min_trap) {
                                lower_range = max_trap;
                                upper_range = new_max - Nt_x;
                            } else if (new_max == max_trap) {
                                lower_range = new_min + Nt_x;
                                upper_range = min_trap;
                            }
                            for (int atom_idx = lower_range;
                                 atom_idx < upper_range; atom_idx += Nt_x) {
                                if (atoms[atom_idx] == 1) {
#ifdef DEBUG_BATCHING_ARO
                                    std::cout << "LOWER RANGE: " << lower_range
                                              << " UPPER RANGE: " << upper_range
                                              << std::endl;
                                    std::cout << "DOWNWARD MIN: " << new_min
                                              << " ATOM_IDX: " << atom_idx
                                              << " NEW_MAX: " << new_max
                                              << std::endl;
                                    std::cout
                                        << "DOWNWARD DID NOT MAKE MOVE: MIN: "
                                        << new_min << " ATOM_IDX: " << atom_idx
                                        << " NEW_MAX: " << new_max << std::endl;
#endif
                                    moving_extra_atom = true;
                                    break;
                                }
                            }
                        }
                    } else if (single_atom_dir[i] < 0) // dest < src -> upward
                        if (new_min != min_trap || new_max != max_trap) {
                            int lower_range = 0;
                            int upper_range = 0;
                            if (new_min == min_trap) {
                                lower_range = max_trap + Nt_x;
                                upper_range = new_max;
                            } else if (new_max == max_trap) {
                                lower_range = new_min + Nt_x + Nt_x;
                                upper_range = min_trap + Nt_x;
                            }
                            for (int atom_idx = lower_range;
                                 atom_idx < upper_range; atom_idx += Nt_x) {
                                if (atoms[atom_idx] == 1) {
#ifdef DEBUG_BATCHING_ARO
                                    std::cout << "LOWER RANGE: " << lower_range
                                              << " UPPER RANGE: " << upper_range
                                              << std::endl;
                                    std::cout << "UPWARD MIN: " << new_min
                                              << " ATOM_IDX: " << atom_idx
                                              << " NEW_MAX: " << new_max
                                              << std::endl;
                                    std::cout
                                        << "UPWARD DID NOT MAKE MOVE: MIN: "
                                        << new_min << " ATOM_IDX: " << atom_idx
                                        << " NEW_MAX: " << new_max << std::endl;
#endif
                                    moving_extra_atom = true;
                                    break;
                                }
                            }
                        }
                    if (moving_extra_atom)
                        break; // breka out if we are moving an extra atom
                    min_trap = std::min(min_trap, std::min(single_atom_start[j],
                                                           single_atom_end[j]));
                    max_trap = std::max(max_trap, std::max(single_atom_start[j],
                                                           single_atom_end[j]));
                    ++j;
                }
            }

            /// update existing locations of atoms
            for (int k = i; k < j; k++) {
                atoms[single_atom_start[k]] = 0;
                atoms[single_atom_end[k]] = 1;
            }

            int offset = min_trap % Nt_x;
            int index = min_trap / Nt_x;
            Synthesis::WfMoveType displacement_page;

            if (horizontal) {
                displacement_page = (single_atom_dir[i] == 1)
                                        ? Synthesis::RIGHT_2D
                                        : Synthesis::LEFT_2D;

                ret.emplace_back(Synthesis::EXTRACT_2D, index,
                                 single_atom_start[i] % Nt_x, 1, 0);
                ret.emplace_back(displacement_page, index, offset, 1, 0);
                ret.emplace_back(Synthesis::IMPLANT_2D, index,
                                 single_atom_end[i] % Nt_x, 1, 0);
                ++i;
                continue;
            }
            bool up = (single_atom_dir[i] > 0);
            displacement_page = up ? Synthesis::UP_2D : Synthesis::DOWN_2D;
            int extraction_extent = up ? max_trap / Nt_x + 1 : Nt_y - index;
            ret.emplace_back(Synthesis::EXTRACT_2D, up ? 0 : index, offset,
                             extraction_extent,
                             0); // block_size == extraction extent here
            /// it may seem like this is turning the time complexity of the
            /// translation into quadratic however, it's really just reordering
            /// the moves returned by the algorithm it's quadratic w.r.t. size
            /// of single_atom_* but still linear w.r.t. size of _src & _dst
            while (true) {
                min_trap = Nt_x * Nt_y;
                max_trap = -1;
                for (int k = i; k < j; ++k) {
                    if (single_atom_start[k] != single_atom_end[k]) {
                        min_trap = std::min(min_trap,
                                            std::min(single_atom_start[k],
                                                     single_atom_start[k] +
                                                         single_atom_dir[i]));
                        max_trap = std::max(max_trap,
                                            std::max(single_atom_start[k],
                                                     single_atom_start[k] +
                                                         single_atom_dir[i]));
                        single_atom_start[k] += single_atom_dir[i];
                    } else {
                        if ((min_trap == Nt_x * Nt_y) && (max_trap == -1)) {
                            continue;
                        } else {
                            break;
                        }
                    }
                }
                if (max_trap == -1)
                    break;
                int blk_size = (max_trap - min_trap) / Nt_x;
                ret.emplace_back(displacement_page, min_trap / Nt_x, offset,
                                 blk_size, extraction_extent);
            }
            ret.emplace_back(Synthesis::IMPLANT_2D, up ? 0 : index, offset,
                             extraction_extent,
                             0); // block_size == extraction_extent here
            i = j;
        }
#ifdef LOGGING_RUNTIME
        p_collector->end_timer("III-Batching", trial_num, rep_num, cycle_num);
#endif
        break;
    }
    default:
        throw std::invalid_argument("Algorithm not supported (gen_moves_list)");
    }

    return ret; // return the vector tuple of moves in the solution moveset
}

/**
 * @brief Resets the Vectors used in the solver based on the dimensions of the
 * traps
 *
 */
void Reconfig::Solver::reset() {
    src.clear();
    dst.clear();
    blk.clear();
    batch_ptrs.clear();
    matching_src.resize(num_atoms_target, 0);
    matching_dst.resize(num_atoms_target, 0);
    src.resize(Nt_x * Nt_x * Nt_y * Nt_y, 0);
    dst.resize(Nt_x * Nt_x * Nt_y * Nt_y, 0);
    blk.resize(Nt_x * Nt_x * Nt_y * Nt_y, 0);
    batch_ptrs.resize(Nt_x * Nt_x * Nt_y * Nt_y, 0);
    path_system.resize(Nt_x * Nt_x * Nt_y * Nt_y, 0);
    path_length.resize(Nt_x * Nt_y, 0);
}

#ifdef IS_SO
extern "C" void solver_wrapper(char *algo_s, int Nt_x, int Nt_y, int *init,
                               int *target, int *result, int *sol_len) {
    Reconfig::Algo algo = get_algo_enum(std::string(algo_s));

    std::vector<int32_t> current_config(init, init + Nt_x * Nt_y);
    std::vector<int32_t> target_config(target, target + Nt_x * Nt_y);

    Reconfig::Solver solver(Nt_x, Nt_y, nullptr);

    float batching_time = 0;
    solver.start_solver(algo, current_config, target_config, 0, 0, 0);

    std::vector<Reconfig::Move> moves_list =
        solver.gen_moves_list(algo, 0, 0, 0);
    *sol_len = moves_list.size();
    std::cout << *sol_len << std::endl;

    for (int i = 0; i < moves_list.size(); ++i) {
        result[i * 4] = (int)std::get<0>(moves_list[i]);     /// Move_type
        result[i * 4 + 1] = (int)std::get<1>(moves_list[i]); /// Index
        result[i * 4 + 2] = (int)std::get<2>(moves_list[i]); /// Offset
        result[i * 4 + 3] = (int)std::get<3>(moves_list[i]); /// Block_size
    }
}

#endif
