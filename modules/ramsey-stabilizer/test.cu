#include <iostream>
#include <fstream>
#include <cmath>
#include <thread>
#include "awg.hpp"

int main() {
    std::cout << "Reading configs" << std::endl;
    AWG awg("iqmixer.yml");

    std::cout << "Openning connection" << std::endl;
    awg.open_connection();

    int segment_size = 6240;
    
    std::cout << "Getting transfer buffer" << std::endl;
    auto transfer_buffer = awg.allocate_transfer_buffer(segment_size);
    // auto copy_buffer = awg.allocate_transfer_buffer(segment_size);
    
    std::cout << "Segment initialization" << std::endl;
    awg.init_segment(0, segment_size);
    awg.init_segment(1, segment_size);
    awg.seqmem_update(0, 0, 1, 0, SPCSEQ_ENDLOOPALWAYS);
    awg.seqmem_update(1, 0, 1, 0, SPCSEQ_ENDLOOPALWAYS);
    
    std::vector<short> vector1(segment_size);
    std::vector<int8> vector2(segment_size);

    // Fill the vectors with a sine wave of frequency 512, amplitude 200
    double frequency = 100.0e3;
    double amplitude = 0x7fff;
    double sample_rate = awg.get_sample_rate();
    for (size_t i = 0; i < segment_size; ++i) {
        double t = i / sample_rate; // Time step
        vector1[i] = static_cast<short>(amplitude * std::sin(2 * M_PI * frequency * t));
        vector2[i] = (vector1[i] >= 0 ? 1 : 0);
    }

    std::cout << "Interleaving" << std::endl;
    awg.interleave_data(*transfer_buffer, {vector1, vector1}, {vector2});

    std::cout << "Segment memory upload" << std::endl;
    awg.load_data(0, *transfer_buffer, segment_size, true);

    std::cout << "Streaming" << std::endl;
    awg.start_stream();

    while (true) {
        std::string input;
        std::cin >> input;
        if (input == "quit") {
            break;
        }
    }

    awg.generate_async_output_pulse(TriggerType::X0);

    std::cout << "Printing awg errors" << std::endl;

    awg.print_awg_error();
    
    awg.close_card();

    return 0;
}