#ifndef LLRS_H_
#define LLRS_H_

#include "Collector.h"
#include "ImageProcessor.h"
#include "JsonWrapper.h"
#include "Sequence.h"
#include "Setup.h"
#include "Solver.h"
#include "WaveformRepo.h"
#include "WaveformTable.h"
#include "acquisition.h"
#include "awg.hpp"
#include "llrs-lib/PreProc.h"
#include "llrs-lib/Settings.h"
#include "utility.h"
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <vector>

enum Target { CENTRE_COMPACT };

class LLRS {
    int cycle_num;
    double detection_threshold;

    std::unique_ptr<Stream::Sequence> awg_sequence;
    std::shared_ptr<Acquisition::ImageAcquisition> image_acquisition;
    std::shared_ptr<Processing::ImageProcessor> img_proc_obj;
    std::shared_ptr<Reconfig::Solver> solver;
    Synthesis::WaveformTable wf_table;
    std::ofstream log_out;
    std::streambuf *old_rdbuf;
    Util::JsonWrapper user_input;
    Reconfig::Algo algo;
    std::vector<int> target_config;
    bool _2d = false;
    int num_trap;

  public:
    class Metadata {
        int Nt_x;
        int Nt_y;
        int num_cycles;
        std::vector<std::vector<Reconfig::Move>> moves_per_cycle;
        std::vector<std::vector<int32_t>> atom_configs;
        std::vector<std::vector<std::tuple<std::string, long long>>>
            runtime_data;
        bool target_met = false;

      public:
        int getNtx() const { return Nt_x; }
        int getNty() const { return Nt_y; }
        int getNumCycles() const { return num_cycles; }
        bool getTargetMet() const { return target_met; }
        const std::vector<std::vector<Reconfig::Move>> &
        getMovesPerCycle() const {
            return moves_per_cycle;
        }
        const std::vector<std::vector<int32_t>> &getAtomConfigs() const {
            return atom_configs;
        }
        const std::vector<std::vector<std::tuple<std::string, long long>>> &
        getRuntimeData() const {
            return runtime_data;
        }
        void reset() {
            Nt_x = 0;
            Nt_y = 0;
            num_cycles = 0;
            target_met = false;
            moves_per_cycle.clear();
            atom_configs.clear();
            runtime_data.clear();
            moves_per_cycle.reserve(ATTEMPT_LIMIT);
            atom_configs.reserve(ATTEMPT_LIMIT);
            runtime_data.reserve(ATTEMPT_LIMIT);
        }

      private:
        // Setter functions
        void setNtx(const int Ntx) { Nt_x = Ntx; }
        void setNty(const int Nty) { Nt_y = Nty; }
        void setNumCycles(const int cycles) { num_cycles = cycles; }
        void setMovesPerCycle(
            const std::vector<std::vector<Reconfig::Move>> &moves) {
            moves_per_cycle = moves;
        }
        void addMovesPerCycle(std::vector<Reconfig::Move> moves) {
            moves_per_cycle.push_back(moves);
        }
        void setAtomConfigs(const std::vector<std::vector<int32_t>> &configs) {
            atom_configs = configs;
        }
        void addAtomConfigs(const std::vector<int32_t> &atom_config) {
            atom_configs.push_back(atom_config);
        }
        void addRuntimeData(
            std::vector<std::tuple<std::string, long long>> cycleRuntimedata) {
            runtime_data.push_back(cycleRuntimedata);
        }
        void incrementNumCycles() { num_cycles++; }
        void setTargetMet() { target_met = true; }
        friend class LLRS;
    } metadata;

    /**
     * @brief Construct a new LLRS object
     *
     * @tparam Args list of arguments that must include the unique or shared
     * pointers correspoding to the AWG, Image Acquisition, Image Processor, and
     * solver objects
     */
    LLRS(std::shared_ptr<AWG> awg = nullptr,
         std::shared_ptr<Acquisition::ImageAcquisition> img_acq = nullptr,
         std::shared_ptr<Processing::ImageProcessor> img_proc = nullptr,
         std::shared_ptr<Reconfig::Solver> solver = nullptr);
    void clean();
    ~LLRS() { clean(); }
    void setup(std::string json_input, bool setup_idle_segment,
               int llrs_step_offset);
    void reset_psf(std::string psf_file);
    void reset_waveform_table();
    void reset_problem(std::string algorithm, int num_target);
    void reset_awg(bool setup_idle_segment, int llrs_step_off) {
        awg_sequence->setup(setup_idle_segment, llrs_step_off, _2d,
                            metadata.getNtx(), metadata.getNty());
    }
    int execute();
    void reset(bool reset_segments);
    std::vector<int32_t> get_target_config(Target target, int num_target);
    void create_center_target(std::vector<int32_t> &target_config,
                              int num_target);
    void setTargetConfig(std::vector<int> new_target_config);
    void store_moves();
    const Metadata &getMetadata() const { return metadata; };
    void get_idle_wfm(typename AWG::TransferBuffer &tb,
                      size_t samples_per_segment) {
        awg_sequence->get_static_wfm(
            *tb, samples_per_segment / awg_sequence->get_waveform_length(),
            metadata.getNtx() * metadata.getNty());
    }
};

#endif
