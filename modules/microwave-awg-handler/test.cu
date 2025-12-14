#include "microwave-awg-handler.h"
#include "microwave-waveforms.h"
#include <iostream>
#include <chrono>
#include <thread>

using namespace MicrowaveHandler;
using namespace MicrowaveWaveforms;

int main() {
    MicrowaveAwgHandler awg_handler("default.yml", "iqmixer.yml");
    awg_handler.open_connection();

    size_t N = 1;

    for (size_t i = 0; i < N; ++i) {
        awg_handler.upload_waveforms({
            SquarePulse(5e3, 100e-6, 0, 0.5),
            Pause(1e-3),
            SquarePulse(5e3, 100e-6, 0, 0.5)
        });
    }

    awg_handler.start();
    
    std::this_thread::sleep_for(std::chrono::microseconds(150));

    std::cout << "Current step: " << awg_handler.get_awg_step() << std::endl;
    awg_handler.awg.print_awg_error();
    
    awg_handler.close_connection();
    
    return 0;
}
    