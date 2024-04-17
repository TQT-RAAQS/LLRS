#include "awg.hpp"
#include "llrs.h"
#include "log.h"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

const int llrs_idle_seg = 1;
const int llrs_idle_step = 1;

void delay(float timeout) {

    auto startTime = std::chrono::high_resolution_clock::now();
    auto targetDuration = std::chrono::duration<float, std::micro>(timeout);

    while (true) {

        if (timeout != -1 &&
            std::chrono::high_resolution_clock::now() - startTime >=
                targetDuration) {
            break;
        }
    }
}

template <typename AWG_T>
void executeLLRS(LLRS<AWG_T> *l, std::shared_ptr<AWG_T> awg) {

    int current_step = awg->get_current_step();
    l->execute();
    l->reset();
    INFO << "LLRS::Execute Completed." << std::endl;
    assert(current_step == llrs_idle_step);
    awg->seqmem_update(llrs_idle_step, llrs_idle_seg, 1, 0,
                       SPCSEQ_ENDLOOPALWAYS);
    delay(awg->get_waveform_duration() * awg->get_waveforms_per_segment() *
          1e6);
    current_step = awg->get_current_step();
    assert(current_step == 0);

    // Ensure LLRS Idle is pointing to itself
    awg->seqmem_update(llrs_idle_step, llrs_idle_seg, 1, llrs_idle_step,
                       SPCSEQ_ENDLOOPALWAYS);
    delay(awg->get_waveform_duration() * awg->get_waveforms_per_segment() *
          1e6);
}

template <typename AWG_T>
void runLLRSOnTrigger(LLRS<AWG_T> *l, std::shared_ptr<AWG_T> awg) {

    int current_step;
    while (true) {

        current_step = awg->get_current_step();
        if (current_step == llrs_idle_seg) {
            executeLLRS(l, awg);
        }
    }
}

template <typename AWG_T>
void streamAWG(LLRS<AWG_T> *l, std::shared_ptr<AWG_T> awg, bool flag_1D,
               std::string problem_config) {

    int16 *pnData = nullptr;

    awg->setup_segment_data(pnData);

    l->setup(problem_config, llrs_idle_seg, llrs_idle_step);

    if (flag_1D == true) {
        l->get_1d_static_wfm(pnData);
    }

    // Load Segment 0 and 1
    awg->init_and_load_range(pnData, awg->get_samples_per_segment(), 0, 1);

    // Segment 0 transitions to segment 1 in sequence memory
    awg->seqmem_update(0, 0, 1, 1, SPCSEQ_ENDLOOPONTRIG);

    // Ensure there is enough time for the first idle segment's pointer to
    // update
    delay(awg->get_waveform_duration() * awg->get_waveforms_per_segment() *
          1e6);

    awg->start_stream();
    awg->print_awg_error();
    assert(awg->get_current_step() == 0);
}
int main(int argc, char *argv[]) {

    // Read problem statement
    std::string problem_id;
    std::string problem_config;
    bool flag_1D = false;

    if (argc > 2) {
        problem_config = std::string(argv[1]);
        flag_1D = (std::string(argv[2]) == "true");

    } else {
        ERROR << " No input was provided" << std::endl;
        return LLRS_ERR;
    }

    std::shared_ptr<AWG> awg{std::make_shared<AWG>()};
    LLRS<AWG> *l = new LLRS<AWG>{awg};
    streamAWG(l, awg, flag_1D, problem_config);
    runLLRSOnTrigger(l, awg);

    return 0;
}
