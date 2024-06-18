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
        std::vector<std::vector<short>> atom_configs;
        nlohmann::json runtime_data;

      public:
        // Getter functions
        const int getNtx() const {return Nt_x;}
        const int getNty() const {return Nt_y;}
        const int getNumCycles() const {return num_cycles;}
        const std::vector<std::vector<Reconfig::Move>> &
        getMovesPerCycle() const {
            return moves_per_cycle;
        }
        const std::vector<std::vector<short>> &getAtomConfigs() const {
            return atom_configs;
        }
        const nlohmann::json &getRuntimeData() const {return runtime_data;}
        const reset() {
            Nt_x = 0;
            Nt_y = 0;
            num_cycles = 0;
            moves_per_cycle.clear();
            atom_configs.clear();
            runtime_data.clear();
            moves_per_cycle.reserve(10);
            atom_configs.reserve(10);
        }

      private:
        Metadata();

        // Setter functions
        void setNtx(const int Ntx) {Nt_x = Ntx;}
        void setNty(const int Nty) {Nt_y = Nty;}
        void setNumCycles(const int cycles) {num_cycles = cycles;}
        void
        setMovesPerCycle(const std::vector<std::vector<Reconfig::Move>> &moves) {
            moves_per_cycle = moves;
        }
        void setAtomConfigs(const std::vector<std::vector<int32_t>> &configs) {atom_configs = configs;}
        void addAtomConfigs(const std::vector<int32_t> &atom_config) {atom_configs.push_back(atom_config);}
        void setRuntimeData(const nlohmann::json &runtimedata) {runtime_data = runtimedata;}
    } metadata;

    LLRS();
    LLRS(std::shared_ptr<AWG_T> &awg);
    void clean();
    ~LLRS() { clean(); }
    void setup(std::string json_input, bool setup_idle_segment,
               int llrs_step_offset, std::string problem_id = "");
    void reset_psf(std::string psf_file);
    int execute();
    void reset(bool reset_segments);
    std::vector<int32_t> get_target_config(Target target, int num_target);
    void create_center_target(std::vector<int32_t> &target_config,
                              int num_target);
    void setTargetConfig(std::vector<int> new_target_config);
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
