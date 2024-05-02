#include <iostream>
#include <fstream>
#include <Setup.h>

int Nt_x = 21;
int Nt_y = 1;
double sample_rate = 624e6;
double waveform_duration = 100e-6;
short waveform_mask = 0x7fff;
int vpp = 0;
std::string waveform_params_address_x = "21_traps.csv", waveform_params_address_y = "21_traps.csv";
std::string output_address = "result.csv";
bool is_transpose = true;

WfType wf_type;
int wf_index = 0, block_size = 0, extraction_extent = 0;

void read_params(int argc, char * argv[]) {
    auto i = 0;

    bool wf_type_flag = false;
    bool wf_index_flag = false;
    bool block_size_flag = false;
    bool extraction_extent_flag = false;

    while (i < argc) {
        if (strcmp(argv[i],"--Nt_x") == 0) {
            Nt_x = std::stoi(argv[++i]);
        } else if (strcmp(argv[i],"--Nt_y") == 0) {
            Nt_y = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--address_x") == 0) {
            waveform_params_address_x = argv[++i];
        } else if (strcmp(argv[i], "--address_y") == 0) {
            waveform_params_address_y = argv[++i];
        } else if (strcmp(argv[i], "--wft") == 0) {
            wf_type = static_cast<WfType>(std::stoi(argv[++i]));
            wf_type_flag = true;
        } else if (strcmp(argv[i], "--index") == 0) {
            wf_index = std::stoi(argv[++i]);
            wf_index_flag = true;
        } else if (strcmp(argv[i], "--block") == 0) {
            block_size = std::stoi(argv[++i]);
            block_size_flag = true;
        } else if (strcmp(argv[i], "--extent") == 0) {
            extraction_extent = std::stoi(argv[++i]);
            extraction_extent_flag = true;
        } else if (strcmp(argv[i], "--address_out") == 0) {
            output_address = argv[++i];
        }

        i++;
    }

    if (!(wf_type_flag && wf_index_flag && block_size_flag && extraction_extent_flag)) {
        throw std::invalid_argument(
            "All the following parameters need to be provided: wf_type, wf_index, block_size, extraction_extent."
            );
    }
}

void save_to_file(short *move_wf_ptr, int length) {
    std::ofstream fout;
    fout.open(output_address);

    fout << move_wf_ptr[0];
    for (int i = 1; i < length; i++) {
        fout << "," << move_wf_ptr[i];
    }

    fout.close();
}

int main(int argc, char * argv[]){
    read_params(argc, argv);
    int waveform_length = sample_rate * waveform_duration;

    auto wf_table = Setup::create_wf_table(
        Nt_x,
        Nt_y,
        sample_rate,
        waveform_duration,
        waveform_length,
        waveform_mask,
        vpp,
        waveform_params_address_x,
        waveform_params_address_y,
        is_transpose
    );
    short *move_wf_ptr = wf_table.get_primary_wf_ptr(wf_type, wf_index, block_size, extraction_extent);
    // short *move_wf_ptr = wf_table.get_primary_wf_ptr(FORWARD, 0, 21, 21);

    save_to_file(move_wf_ptr, waveform_length);
}
