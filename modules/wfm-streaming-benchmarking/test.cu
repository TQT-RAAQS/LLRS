/*
This test is useful for streaming waveforms on both output channels of the AWG.
Make sure to change the configs of the awg to stream on both channels with a
save Vpp and amplitudes.
*/

#include "Waveform.h"
#include "awg.hpp"
#include <iostream>
#include <vector>

std::vector<short> translate(std::vector<double> input) {
    std::vector<short> result;
    result.reserve(input.size());
    for (unsigned i = 0; i < input.size(); i++) {
        result.push_back((short)(input.at(i) * 0x7fff));
    }
    return std::move(result);
}

std::vector<short> interleave(std::vector<short> wf1, std::vector<short> wf2) {
    std::vector<short> result;
    result.reserve(2 * wf1.size());
    for (unsigned i = 0; i < wf1.size(); i++) {
        result.push_back(wf1.at(i));
        result.push_back(wf2.at(i));
    }
    return std::move(result);
}

int main() {
    AWG awg{};
    double wf_duration = 10e-6;
    double sample_rate = awg.get_sample_rate();

    Synthesis::Waveform::set_static_function(
        std::make_unique<Synthesis::Sin>());

    double alpha1 = 1, nu1 = 100e6, phi1 = 0;
    double alpha2 = 0.4, nu2 = 100e6, phi2 = 0;
    std::vector<short> wf1 = translate(
        Synthesis::Idle(wf_duration, std::make_tuple(alpha1, nu1, phi1))
            .discretize(sample_rate));
    std::vector<short> wf2 = translate(
        Synthesis::Idle(wf_duration, std::make_tuple(alpha2, nu2, phi2))
            .discretize(sample_rate));
    std::vector<short> wf = interleave(wf1, wf2);

    AWG::TransferBuffer buffer = awg.allocate_transfer_buffer(
        static_cast<int>(wf_duration * sample_rate * 2), false);

    memcpy(*buffer, wf.data(),
           static_cast<int>(2 * wf_duration * sample_rate * sizeof(short)));
    awg.init_and_load_range(*buffer,
                            static_cast<int>(wf_duration * sample_rate), 0, 1);

    awg.seqmem_update(0, 0, 1, 0, SPCSEQ_ENDLOOPONTRIG);

    awg.start_stream();

    std::this_thread::sleep_for(std::chrono::seconds(2));

    awg.stop_card();

    std::cout << "Execution done." << std::endl;
    return 1;
}