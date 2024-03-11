#ifndef LLRS_H
#define LLRS_H

#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>
#include "LLRS-lib/Settings.h"
#include "Setup.h"
#include "LLRS-lib/Preproc.h"
#include "JsonWrapper.h"
#include "Collector.h"
#include "Solver.h"
#include "WaveformRepo.h"
#include "WaveformTable.h"
#include "Sequence.h"
#include "ImageProcessor.h"
#include "utility.h"
#include "activesilicon-1xcld.hpp"
#include "awg.hpp"
#include <memory>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
using json = nlohmann::json;



/* Toggle for storing all images taken */
// #define STORE_IMAGES
#define LOAD_SINGLE_SEGMENT

enum Target {
    CENTRE_COMPACT
};

template<typename AWG_T> class LLRS{
    int     trial_num;
    int     rep_num;
    int     cycle_num;
    double  detection_threshold;

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

        friend class LLRS;

    public:
        // Getter functions
        const int getNtx() const;
        const int getNty()const;
        const int getNumCycles() const;
        const std::vector<std::vector<Reconfig::Move>>& getMovesPerCycle() const;
        const std::vector<std::vector<int32_t>>& getAtomConfigs() const;

        Metadata(const Metadata& other) {
            Nt_x = other.Nt_x;
            Nt_y = other.Nt_y;
            num_cycles = other.num_cycles;
            moves_per_cycle = other.moves_per_cycle;
            atom_configs = other.atom_configs;
        }
    private:
        Metadata();

        // Setter functions
        void setNtx(const int Ntx);
        void setNty(const int Nty);
        void setNumCycles(const int cycles);
        void setMovesPerCycle(const std::vector<std::vector<Reconfig::Move>>& moves);
        void setAtomConfigs(const std::vector<std::vector<int32_t>>& configs);
        void addAtomConfigs(const std::vector<int32_t>& atom_config);
        } metadata;


    LLRS();
    LLRS(std::shared_ptr<AWG_T> &awg);
    void clean();
    ~LLRS() {clean();}
    void setup(std::string json_input, size_t llrs_seg_offset, size_t llrs_step_offset, std::string problem_id = "");
    void small_setup( std::string json_input );
    void execute();
    void reset();
    std::vector<int32_t> get_target_config(Target target, int num_target );
    void create_center_target ( std::vector<int32_t> &target_config, int num_target );
    void setTargetConfig(std::vector<int> new_target_config);
    void set_psf(std::string json_input);
    void set_target(std::string json_input);
    int getTrialNum();
    int getRepNum();
    void store_moves();
    const Metadata& getMetadata() const {return metadata;};
    void get_1d_static_wfm(int16 *pnData, int num_wfms) {awg_sequence->get_1d_static_wfm(pnData, num_wfms, metadata.getNtx());}
};


#endif
