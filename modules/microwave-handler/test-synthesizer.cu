#include "microwave-waveform-synthesizer.h"
#include <iostream>
#include <chrono>
#include <thread>
#include "awg.hpp"

using namespace MicrowaveHandler;
using namespace MicrowaveHandler;

int main() {
    auto awg = AWG{"iqmixer.yml"};

    int N = 624000'00;
    double t0 = 312e-6;
    double t_initial_pause = 12e-6;
    double duration = 80e-3;
    double phase = 0; // in rad
    double amplitude = 0.004; // in V

    Waveform wf = SquarePulse(
        100e6,
        duration,
        phase,
        amplitude
    );

    awg.open_connection();
    auto buffer1 = awg.allocate_transfer_buffer(N, false);
    auto buffer2 = awg.allocate_transfer_buffer(N, false);
    awg.close_card();

    auto synthesizer = MicrowaveWaveformSynthesizer{};
    synthesizer.set_digital_offset_time(-9.61538462e-08);

    auto before = std::chrono::high_resolution_clock::now();
    synthesizer.generate_pulse(
        *buffer1,
        N,
        awg,
        wf,
        t0,
        t_initial_pause
    );
    auto after = std::chrono::high_resolution_clock::now();
    std::cout << "Method 1: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count() 
              << " ms" << std::endl;

    synthesizer.set_fast_interleaving_flag(true);

    before = std::chrono::high_resolution_clock::now();
    synthesizer.generate_pulse(
        *buffer2,
        N,
        awg,
        wf,
        t0,
        t_initial_pause
    );
    after = std::chrono::high_resolution_clock::now();
    std::cout << "Method 2: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count() 
              << " ms" << std::endl;

    for (int i = 0; i < 2*N; ++i) {
        assert ((*buffer1)[i] == (*buffer2)[i]);
    }
    
    return 0;
}