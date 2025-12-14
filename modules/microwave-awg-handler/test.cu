#include <iostream>
#include "microwave-awg-handler.h"
#include "microwave-waveforms.h"

using namespace MicrowaveHandler;
using namespace MicrowaveWaveforms;

int main() {
    MicrowaveAwgHandler awg_handler("default.yml", "iqmixer.yml");
    
    // auto res = awg_handler.breakdown_waveforms({
    //     SquarePulse(5e3, 2.0e-6, 0, 0.5),
    //     Pause(32e-6),
    //     Pause(35e-6),
    //     SquarePulse(5e3, 20e-6, 0, 0.5),
    // });

    // for (const auto &c : std::get<0>(res)) {
    //     auto s = c.waveform;

    //     if (auto *p = boost::get<Pause>(&s)) {
    //         std::cout << "t=" << c.time*1e6 << " us | PAUSE " << p->duration * 1e6 << " us\n";
    //     }
    //     if (auto *p = boost::get<SquarePulse>(&s)) {
    //         std::cout << "t=" << c.time*1e6 << " us | SIN " << p->duration * 1e6 << " us, delay = " << c.t_initial_pause*1e6 << " us\n";
    //     }
    // }
    // std::cout << "********************\n";
    // for (const auto &c : std::get<1>(res)) {
    //     std::cout << c << std::endl;
    // }

    return 0;
}