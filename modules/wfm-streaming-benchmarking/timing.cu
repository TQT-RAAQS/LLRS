#include "Solver.h"
#include "awg.hpp"
#include <Setup.h>
#include <chrono>
#include <iostream>
#include <string.h>

bool cont_buffer = false;
int wf_per_segment = 32;
int num_rep = 1000;

/**
 * @brief Function that takes command line input and modifies the default global
 * parameters
 */
void cmd_line(int argc, char *argv[]) {
    int i = 1;
    while (i < argc) {
        if (strcmp(argv[i], "-cb") == 0) {
            cont_buffer = true;
        } else if (strcmp(argv[i], "-wfps") == 0) {
            wf_per_segment = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-nr") == 0) {
            num_rep = std::stoi(argv[++i]);
        }
        i++;
    }
}

int main(int argc, char *argv[]) {

    cmd_line(argc, argv);
    AWG awg;

    awg.configure_segment_length(
        Synthesis::read_waveform_duration(WFM_CONFIG_PATH("config.yml")));
    size_t samples_per_segment = wf_per_segment * awg.get_waveform_length();
    double awg_sample_rate = awg.get_sample_rate();
    int waveform_length = awg.get_waveform_length();
    int wfm_mask = awg.get_wavefrom_mask();
    int vpp = awg.get_vpp();
    auto wf_table =
        Setup::create_wf_table(21, 1, awg_sample_rate, wfm_mask, vpp,
                               "21_traps.csv", "21_traps.csv", true);
    auto transfer_buffer =
        awg.allocate_transfer_buffer(samples_per_segment, cont_buffer);
    awg.init_segment(1, samples_per_segment);

    std::vector<double> results;
    results.reserve(num_rep);
    std::vector<Reconfig::Move> moves;
    moves.emplace_back(Synthesis::IDLE_1D, 0, 0, 21, 0);
    for (int i = 0; i < num_rep; ++i) {
        std::chrono::high_resolution_clock::time_point start =
            std::chrono::high_resolution_clock::now();

        for (int wf_idx = 0; wf_idx < wf_per_segment; wf_idx++) {
            auto move = moves[0];
            short *move_wf_ptr = wf_table.get_waveform_ptr(
                std::get<0>(move), std::get<4>(move), std::get<1>(move),
                std::get<2>(move), std::get<3>(move));
            memcpy(*transfer_buffer + wf_idx * awg.get_waveform_length(),
                   move_wf_ptr, awg.get_waveform_length() * sizeof(short));
        }
        awg.load_data(1, *transfer_buffer, samples_per_segment * sizeof(short));

        std::chrono::high_resolution_clock::time_point end =
            std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span =
            std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                      start);
        results.push_back(time_span.count());
    }
    double average =
        std::accumulate(results.begin(), results.end(), 0.0) / num_rep;
    std::vector<double> diffs;
    diffs.reserve(results.size());
    for (auto it : results) {
        diffs.push_back(it - average);
    }
    double stddev = std::sqrt(
        std::inner_product(diffs.begin(), diffs.end(), diffs.begin(), 0.0) /
        (num_rep - 1));
    std::cout << average << std::endl << stddev << std::endl;
}
