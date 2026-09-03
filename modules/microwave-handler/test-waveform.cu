#include "microwave-awg-handler.h"
#include "microwave-waveform-synthesizer.h"
#include <iostream>
#include <chrono>
#include <thread>
#include "awg.hpp"

using namespace MicrowaveHandler;

int main() {
    auto awg = AWG{"iqmixer.yml"};

    int N = 624000;
    double frequency = 90e6;
    double t0 = 0e-6;
    double t_initial_pause = 0;
    double duration = 1e-3;
    double phase = 0; // in rad
    double amplitude = 0.05; // in V

    Waveform wf = SquarePulse(
        frequency,
        duration,
        phase,
        amplitude
    );

    awg.open_connection();
    auto buffer = awg.allocate_transfer_buffer(N, false);
    
    auto synthesizer = MicrowaveWaveformSynthesizer{};
    synthesizer.set_digital_offset_time(0);
    synthesizer.set_fast_interleaving_flag(true);

    synthesizer.generate_pulse(
        *buffer,
        N,
        awg,
        wf,
        t0,
        t_initial_pause
    );

    //////////////////////////////////////////

    auto segment_index = int(0);
    auto step_index = segment_index;

    awg.init_segment(segment_index, N);
    awg.load_data(segment_index, *buffer, N, true);

    awg.set_initial_step(0);
    awg.seqmem_update(
        step_index,
        segment_index,
        1,
        step_index,
        SPCSEQ_ENDLOOPALWAYS
    );

    awg.start_stream();

    std::cout << "Streaming..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::cout << "Done." << std::endl;

    awg.close_card();
    
    return 0;
}