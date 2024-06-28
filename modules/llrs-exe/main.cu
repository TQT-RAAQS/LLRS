#include "awg.hpp"
#include "llrs.h"
#include "log.h"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

const int llrs_idle_seg = 0;
const int llrs_idle_step = 1;

void my_handler(int s) {
    printf("Caught signal %d\n", s);
    exit(1);
}

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

int main(int argc, char *argv[]) {
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    // Read problem statement
    std::string problem_id;
    std::string problem_config;

    if (argc > 1) {
        problem_config = std::string(argv[1]);
    } else {
        ERROR << "No input was provided: Please provide the config file as the "
                 "argument."
              << std::endl;
        return LLRS_ERR;
    }

    std::shared_ptr<AWG> awg{std::make_shared<AWG>()};
    size_t samples_per_idle_segment =
        awg->get_idle_segment_length() * awg->get_waveform_length();

    auto tb = awg->allocate_transfer_buffer(samples_per_idle_segment, false);
    awg->fill_transfer_buffer(tb, samples_per_idle_segment, 0);
    awg->init_and_load_range(*tb, samples_per_idle_segment, 0, 1);

    LLRS l{awg};
    l.setup(problem_config, false, llrs_idle_step);

    std::cout << "Starting AWG stream" << std::endl;

    if (awg->get_idle_segment_wfm()) {
        l.get_idle_wfm(tb, samples_per_idle_segment);
        awg->init_and_load_range(*tb, samples_per_idle_segment, 0, 1);
    }

    // SEQUENCE MEMORY
    awg->seqmem_update(0, 0, 1, 1, SPCSEQ_ENDLOOPONTRIG);

    // Ensure there is enough time for the first idle segment's pointer to
    // update
    float timeout =
        awg->get_waveform_duration() * awg->get_idle_segment_length() * 1e6;
    auto startTime = std::chrono::high_resolution_clock::now();
    auto targetDuration = std::chrono::duration<float, std::micro>(timeout);

    while (true) {

        // Timeout loop break
        if (timeout != -1 &&
            std::chrono::high_resolution_clock::now() - startTime >=
                targetDuration) {
            break;
        }
    }

    awg->start_stream();

    awg->seqmem_update(0, 0, 1, 1, SPCSEQ_ENDLOOPONTRIG);

    // LLRS Execution
    int num_of_executions, num_of_successes = 0;
    bool flag = false;
    while (true) {
        flag = false;
        std::cout << "Waiting for trigger to run LLRS" << std::endl;
        int current_seg = 0;
        auto startTime = std::chrono::high_resolution_clock::now();
        auto targetDuration = std::chrono::seconds(60);
        while (true) {
            // Timeout loop break
            if (std::chrono::high_resolution_clock::now() - startTime >=
                targetDuration) {
                flag = true;
                break;
            }

            current_seg = awg->get_current_step();

            if ((current_seg != 0)) {
                // trigger event occured
                std::cout
                    << "TriggerDetector: trigger event occured - Current Seg: "
                    << current_seg << std::endl;

                std::cout << "Starting the LLRS" << std::endl;
                int success = l.execute();
                std::cout << "Done LLRS::Execute" << std::endl;
                ++num_of_executions;
                if (success == 0)
                    ++num_of_successes;

                current_seg = awg->get_current_step();
                assert(current_seg == llrs_idle_step);
                awg->seqmem_update(llrs_idle_step, llrs_idle_seg, 1, 0,
                                   SPCSEQ_ENDLOOPALWAYS);
                awg->seqmem_update(llrs_idle_step, llrs_idle_seg, 1,
                                   llrs_idle_step, SPCSEQ_ENDLOOPALWAYS);

                l.reset(true);
            }
        }
        if (flag) {
            std::cout << "No Trigger detected, exiting" << std::endl;
            break;
        }
    }

    std::cout << "Number of successful LLRS executions: " << num_of_successes
              << " out of " << num_of_executions << std::endl;
    std::cout << "Program terminated gracefully.\n";

    return 0;
}
