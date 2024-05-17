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
#include "activesilicon-1xcld.hpp"
#include "awg.hpp"
#include "llrs-lib/PreProc.h"
#include "llrs-lib/Settings.h"
#include "utility.h"
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>
#include <yaml-cpp/yaml.h>
using json = nlohmann::json;

/* Toggle for storing all images taken */
// #define STORE_IMAGES

enum Target { CENTRE_COMPACT };

template <typename AWG_T> class LLRS {
    int trial_num;
    int rep_num;
    int cycle_num;
    double detection_threshold;

    Util::Collector *p_collector;
    Synthesis::WaveformTable wf_table;
    std::unique_ptr<Stream::Sequence<AWG_T>> awg_sequence;
    std::unique_ptr<Acquisition::ActiveSilicon1XCLD> fgc;
    Processing::ImageProcessor img_proc_obj;
    Reconfig::Solver solver;
    std::ofstream log_out;
    std::streambuf *old_rdbuf;
    Util::JsonWrapper user_input;
    Reconfig::Algo algo;
    std::vector<int> target_config;
    bool _2d = false;
    int num_trap;
    std::string problem_id;

  public:
    class Metadata {
        int Nt_x;
        int Nt_y;
        int num_cycles;
        std::vector<std::vector<Reconfig::Move>> moves_per_cycle;
        std::vector<std::vector<int32_t>> atom_configs;
        nlohmann::json runtime_data;

        friend class LLRS;

      public:
        // Getter functions
        const int getNtx() const;
        const int getNty() const;
        const int getNumCycles() const;
        const std::vector<std::vector<Reconfig::Move>> &
        getMovesPerCycle() const;
        const std::vector<std::vector<int32_t>> &getAtomConfigs() const;
        const nlohmann::json &getRuntimeData() const;

        Metadata(const Metadata &other) {
            Nt_x = other.Nt_x;
            Nt_y = other.Nt_y;
            num_cycles = other.num_cycles;
            moves_per_cycle = other.moves_per_cycle;
            atom_configs = other.atom_configs;
            runtime_data = other.runtime_data;
        }

      private:
        Metadata();

        // Setter functions
        void setNtx(const int Ntx);
        void setNty(const int Nty);
        void setNumCycles(const int cycles);
        void
        setMovesPerCycle(const std::vector<std::vector<Reconfig::Move>> &moves);
        void setAtomConfigs(const std::vector<std::vector<int32_t>> &configs);
        void addAtomConfigs(const std::vector<int32_t> &atom_config);
        void setRuntimeData(const nlohmann::json &runtime_data);
    } metadata;

    LLRS();
    LLRS(std::shared_ptr<AWG_T> &awg);
    void clean();
    ~LLRS() { clean(); }
    void setup(std::string json_input, bool setup_idle_segment,
               int llrs_step_offset, std::string problem_id = "");
    void small_setup(std::string json_input);
    int execute();
    void reset(bool reset_segments);
    std::vector<int32_t> get_target_config(Target target, int num_target);
    void create_center_target(std::vector<int32_t> &target_config,
                              int num_target);
    void setTargetConfig(std::vector<int> new_target_config);
    int getTrialNum();
    int getRepNum();
    void store_moves();
    const Metadata &getMetadata() const { return metadata; };
    void get_idle_wfm(typename AWG_T::TransferBuffer &tb,
                      size_t samples_per_segment) {
        awg_sequence->get_static_wfm(
            *tb, samples_per_segment / awg_sequence->get_waveform_length(),
            metadata.getNtx() * metadata.getNty());
    }
};

#endif
