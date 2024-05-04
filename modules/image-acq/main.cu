#include "activesilicon-1xcld.hpp"
#include "awg.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <unistd.h>
#include <string>

int roi_y = 1024;
int roi_x = 1024;

int hor_bin = 1;
int ver_bin = 1;

int fgc_timeout = 600; // ms

int reps = 10;

/**
* @brief Function that takes command line input and modifies the default ROI parameters
*/
void cmd_line(int argc, char * argv[]) 
{
	int i = 1;
	while (i < argc)
	{
		if (strcmp(argv[i],"-rx") == 0){
			roi_x = std::stoi(argv[++i]);
		} else if (strcmp(argv[i], "-ry") == 0) {
			roi_y = std::stoi(argv[++i]);
		} else if (strcmp(argv[i], "-hb") == 0) {
			hor_bin = std::stoi(argv[++i]);
		} else if (strcmp(argv[i], "-vb") == 0) {
			ver_bin = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-fgct") == 0) {
            fgc_timeout = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0) {
            reps = std::stoi(argv[++i]);
        }

        i++;
    }
}

int main(int argc, char * argv[]){

    // Take command line input to change paramaters as necessary
    cmd_line(argc, argv);

    std::unique_ptr<AWG> awg = std::make_unique<AWG>();
    std::unique_ptr<Acquisition::ActiveSilicon1XCLD> fgc = std::make_unique<Acquisition::ActiveSilicon1XCLD>(roi_x, roi_y, fgc_timeout, 0, 0, ver_bin, hor_bin);


    std::vector<std::string> time_list;

    double acq_time = 0;
    for (int i = 0; i < reps; i++) {

        std::chrono::steady_clock::time_point begin;
        std::chrono::steady_clock::time_point end;
        
        awg->generate_async_output_pulse(EMCCD);
        begin = std::chrono::steady_clock::now();
        std::vector<uint16_t> image  = fgc->acquire_single_image();
        end = std::chrono::steady_clock::now();

	// Append the time to list of times to be returned
        time_list.push_back(std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()));

    }

    for (std::string x: time_list){
        std::cout << x << ",";
    }
    std::cout << std::endl;
}
