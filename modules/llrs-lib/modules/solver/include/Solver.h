#ifndef SOLVER_H_
#define SOLVER_H_
#include "Algorithms.h"
#include "Collector.h"
#include "WaveformTable.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <llrs-lib/Settings.h>
#include <numeric>

/**
 * @brief The Reconfig namespace contains all the classes and functions related
 * to the reconfiguration problem
 *
 */
namespace Reconfig {

/**
 * @brief The tuple representing a move
 * @var WfMoveType The type of the Waveform Movement
 * @var int index
 * @var int offset  (0 if 1D)
 * @var int block size
 * @var int extrtaction extent
 */
using Move = std::tuple<Synthesis::WfMoveType, int, int, int, int>;

/**
 * @brief The Algo Enum representing the Algorithm to be used for the
 * reconfiguration problem
 */
enum Algo {
    LINEAR_EXACT_1D,
    LINEAR_EXACT_V2_1D,
    LINEAR_EXACT_V2_GPU_1D,
    REDREC_CPU_V2_2D,
    REDREC_CPU_V3_2D,
    REDREC_GPU_V3_2D,
    BIRD_CPU_2D,
    ARO_CPU_2D
};

// -------------------------------------------------------------------------------

/**
 * @brief The Solver class acts an abstraction layer (wrapper) for the algorithm
 * solving
 *
 */
class Solver {
    int Nt_x = 0;
    int Nt_y = 0;
    int num_atoms_initial = 0;
    int num_atoms_target = 0;
    int wfm_per_segment;
    std::vector<int32_t> matching_src;
    std::vector<int32_t> matching_dst;
    std::vector<int32_t> src;
    std::vector<int32_t> dst;
    std::vector<int32_t> blk;
    std::vector<int32_t> batch_ptrs;
    std::vector<int32_t> path_system;
    std::vector<int32_t> path_length;
    std::vector<int32_t> initial;

  public:
    /**
     * @brief Setup a new Solver object
     *
     * @param Nt_x
     * @param Nt_y
     * @param num_targets
     */
    void setup(int Nt_x, int Nt_y, int wfm_per_segment);

    /**
     * @brief The unified Solver method for 1D and 2D
     * For 1D, the process is equivalent to calling start_matching_1d followed
     * by calling start_batching_1d
     * @param algo_select
     * @param initial
     * @param target
     * @param batching_time_ptr
     * @return true
     * @return false
     */
    bool start_solver(Algo algo_select, std::vector<int32_t> initial,
                      std::vector<int32_t> target);

    /**
     *   @brief Generate list of moves for reconfiguration algorithm
     *   @param algo_select An enum value specifying which reconfiguration
     * algorithm to use
     *   @return A vector of Move tuples representing the primary waveform
     * keying components
     */
    std::vector<Move> gen_moves_list(Algo algo_select);

    /**
     * @brief Resets the Vectors used in the solver based on the dimensions of
     * the traps
     *
     */
    void reset();

    /**
     * @brief Get the src atoms vector
     *
     */
    std::vector<int32_t> get_src_atoms() const { return src; };

    /**
     * @brief Get the dst atoms vector
     *
     */
    std::vector<int32_t> get_dst_atoms() const { return dst; };

    /**
     * @brief Get the blk_sizes vector
     *
     */
    std::vector<int32_t> get_blk_sizes() const { return blk; };

    /**
     * @brief Get the batch_ptrs vector
     *
     */
    std::vector<int32_t> get_batch_ptrs() const { return batch_ptrs; };
};
} // namespace Reconfig

#endif

extern "C" void solver_wrapper_2d(char *algo_s, int Nt_x, int Nt_y, int *init,
                                  int *target, int *result, int *sol_len);
extern "C" void solver_wrapper_1d(char *algo_s, int Nt_y, int *init,
                                  int *target, int *result, int *sol_len);
