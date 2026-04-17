#include "microwave-waveforms.h"

using namespace MicrowaveHandler;

Waveform MicrowaveHandler::waveform_from_string(const std::string& str) {
    std::istringstream iss(str);
    std::string type;
    iss >> type;

    if (type == "PAUSE") {
        double duration;
        if (!(iss >> duration)) {
            throw std::runtime_error("Invalid PAUSE waveform string");
        }
        return Pause(duration);
    }

    if (type == "SQUARE") {
        double freq, dur, phase, amp;
        if (!(iss >> freq >> dur >> phase >> amp)) {
            throw std::runtime_error("Invalid SQUARE waveform string");
        }
        return SquarePulse(freq, dur, phase, amp);
    }

    if (type == "SQUARE60") {
        double freq, dur, phase, amp;
        if (!(iss >> freq >> dur >> phase >> amp)) {
            throw std::runtime_error("Invalid SQUARE60 waveform string");
        }
        return Square60Pulse(freq, dur, phase, amp);
    }

    throw std::runtime_error("Unknown waveform type: " + type);
}