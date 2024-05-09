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

const int llrs_idle_seg = 1;
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
    bool flag_1D = true;

    if (argc > 2) {
        problem_config = std::string(argv[1]);
        flag_1D = (std::string(argv[2]) == "true");
    } else if (argc > 1) {
        problem_config = std::string(argv[1]);
    } else {
        ERROR << "No input was provided: Please provide the config file as the "
                 "argument."
              << std::endl;
        return LLRS_ERR;
    }

    std::shared_ptr<AWG> awg{std::make_shared<AWG>()};
    LLRS<AWG> *l = new LLRS<AWG>{awg};
    int16 *pnData = nullptr;
    const int64_t samples_per_segment = awg->get_samples_per_segment();

    l->setup(problem_config, llrs_idle_seg, llrs_idle_step);
    l->get_1d_static_wfm(pnData);
    awg->seqmem_update(0, 0, 1, 1, SPCSEQ_ENDLOOPONTRIG);

    awg->start_stream();
    awg->print_awg_error();
    assert(awg->get_current_step() == 0);

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
                int success = l->execute();
                l->reset();
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
