#include "microwave-awg-handler.h"
#include "microwave-waveforms.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <awg.hpp>

using namespace MicrowaveHandler;
using namespace MicrowaveWaveforms;

int main() {
    MicrowaveAwgHandler awg_handler("default.yml");
    awg_handler.open_connection();

    awg_handler.upload_waveforms({
        Pause(500e-3),
        SquarePulse(99356555.0, 500e-6, 0, 0.042),
    });

    awg_handler.start();
    
    while (true) {
        auto s = awg_handler.get_awg_step();
        std::cout << awg_handler.get_awg_step() << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (s == 1) {
            awg_handler.stop();
            awg_handler.upload_waveforms({
                Pause(500e-3),
                SquarePulse(99356555.0, 500e-6, 0, 0.042),
            });
            awg_handler.start();
        }
    }

    awg_handler.close_connection();
    
    return 0;
}
    