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

    int segment_size = 624000;
    
    std::cout << "Getting transfer buffer" << std::endl;
    auto transfer_buffer = awg.allocate_transfer_buffer(segment_size);
    // auto copy_buffer = awg.allocate_transfer_buffer(segment_size);
    
    std::cout << "Segment initialization" << std::endl;
    
    std::vector<short> vector1(segment_size);
    std::vector<int8> vector2(segment_size);

    double frequency = 100.0e6;
    double amplitude = 0x7fff;
    double sample_rate = awg.get_sample_rate();

    auto start = std::chrono::high_resolution_clock::now();

    awg.init_segment(0, segment_size);
    awg.init_segment(1, segment_size);
    awg.seqmem_update(0, 0, 1, 0, SPCSEQ_ENDLOOPALWAYS);
    awg.seqmem_update(1, 0, 1, 0, SPCSEQ_ENDLOOPALWAYS);

    #pragma omp simd
    for (size_t i = 0; i < segment_size; ++i) {
        double t = i / sample_rate; // Time step
        vector1[i] = static_cast<short>(amplitude * std::sin(2 * M_PI * frequency * t));
        vector2[i] = 1;
    }
    awg.interleave_data(*transfer_buffer, {vector1, vector1}, {vector2});
    awg.load_data(0, *transfer_buffer, segment_size, true);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Function took " << duration << " us" << std::endl;


    std::cout << "Streaming" << std::endl;
    awg.start_stream();

    while (true) {
        std::string input;
        std::cin >> input;
        if (input == "quit") {
            break;
        }
    }

    std::cout << "Printing awg errors" << std::endl;

    awg.print_awg_error();
    
    awg.close_card();

    return 0;
}