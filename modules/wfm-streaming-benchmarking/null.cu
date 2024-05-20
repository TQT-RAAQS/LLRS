#include "Sequence.h"
#include "Setup.h"
#include "Solver.h"
#include "WaveformTable.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>

int number_of_moves = 100;
int number_of_waveforms_per_segment = 32;
int Nt_y = 1;
int Nt_x = 16;
int num_rep = 1000;

std::vector<Reconfig::Move> generate_moves(int number_of_moves);

int main(int argc, char *argv[]) { // <Number of moves>, <numbers of waveforms
                                   // per segment>, <num of reps>

    if (argc > 3) {
        number_of_moves = std::stoi(argv[1]);
        number_of_waveforms_per_segment = std::stoi(argv[2]);
        num_rep = std::stoi(argv[3]);
    } else if (argc > 2) {
        number_of_moves = std::stoi(argv[1]);
        number_of_waveforms_per_segment = std::stoi(argv[2]);
    } else if (argc > 1) {
        number_of_moves = std::stoi(argv[1]);
    }

    {
        std::string problem_config =
            std::string("") + "# AWG Global Settings \n \
        driver_path:              /dev/spcm0    # the device file system handle : string \n \
        external_clock_freq:      10000000      # Hz : int \n \
        channels:                 [0]           # list of channels : int[] \n \
        amp:                      [140]         # list of amp for each channel in mV : int[] \n \
        # Segment Settings \n \
        awg_num_segments:         256           # number of segments : int \n \
        sample_rate:              624e6         # sample rate : float  \n \
        wfm_mask:                 0x00007fff    # waveform mask : Hex   \n \
        waveform_duration:        10e-6        # waveform duration : float  \n \
        waveforms_per_segment:    " +
            std::to_string(number_of_waveforms_per_segment) +
            "      # waveforms per segment : int  \n \
        null_segment_length:      6240         # null segment length : int \n \
        idle_segment_length:      6240         # idle segment length : int \n \
        idle_segment_wfm:         true          # whether or not to fill the IDLE segment with static wfms : bool \n \
        # Trigger Settings \n \
        trigger_size:             2000          # number of Waveforms repeated for the synchronized trigger: int \n \
        vpp:                      140           # peak to peak voltage in mV : int \n \
        # EMCCD Trigger Settings \n \
        acq_timeout:              600           # acquisition timeout wait time for the EMCCD response in ms : int \n \ 
		async_trig_amp:           3             # trigger Amp in volts : int \n";
        std::ofstream config_file{std::string("") + PROJECT_BASE_DIR +
                                  "/configs/awg/awg.yml"};
        config_file << problem_config;
    }
    Synthesis::WaveformTable wf_table;
    Stream::Sequence<AWG> awg_sequence{nullptr, wf_table};

    double awg_sample_rate = awg_sequence.get_sample_rate();
    double waveform_duration = awg_sequence.get_waveform_duration();
    int waveform_length = awg_sequence.get_waveform_length();
    int wfm_mask = awg_sequence.get_waveform_mask();
    int vpp = awg_sequence.get_vpp();

    wf_table = Setup::create_wf_table(
        Nt_x, Nt_y, awg_sample_rate, waveform_duration, waveform_length,
        wfm_mask, vpp, "21_traps.csv", "21_traps.csv", true);
    awg_sequence.setup(true, 0, false, Nt_x, Nt_y);
    awg_sequence.start_stream();

    int counter = 0;
    for (int i = 0; i < num_rep; i++) {
        auto moves = generate_moves(number_of_moves);
        counter += awg_sequence.load_and_stream(moves, 0, 0, 0);
        awg_sequence.reset(false);
    }
    std::cout << (float)counter / num_rep << std::endl;
}

std::vector<Reconfig::Move> generate_moves(int number_of_moves) {
    std::vector<Reconfig::Move> moves;
    for (int i = 0; i < number_of_moves;
         i++) { // this function can be changed to generate random moves, even
                // though it will not affect timing or the result of the
                // benchmark
        moves.emplace_back(Synthesis::IDLE_1D, 0, 0, 1, Nt_x * Nt_y);
    }
    return moves;
}
