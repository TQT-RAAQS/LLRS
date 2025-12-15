#ifndef _MICROWAVE_WAVEFORMS_H_
#define _MICROWAVE_WAVEFORMS_H_

#include <string>
#include <boost/variant.hpp>
#include <sstream>

namespace MicrowaveWaveforms {

    struct Pause;
    struct SquarePulse;

    using Waveform = boost::variant<
        Pause,
        SquarePulse
    >;

    Waveform from_string(const std::string& str);

    struct Pause {
        double duration;   // duration in seconds

        Pause(double dur) : duration(dur) {}

        std::string to_string() const {
            return "PAUSE " + std::to_string(duration);
        }

        int64_t hash() const {
            return std::hash<double>{}(duration);
        }
    };

    struct SquarePulse {
        double detuning;  // in Hz
        double duration;   // in seconds
        double phase;      // in radians
        double amplitude;  // in units of V

        SquarePulse(double freq, double dur, double ph, double amp)
            : detuning(freq), duration(dur), phase(ph), amplitude(amp) {}

        std::string to_string() const {
            return "SQUARE " +
                std::to_string(detuning) + " " +
                std::to_string(duration) + " " +
                std::to_string(phase) + " " +
                std::to_string(amplitude);
        }

        int64_t hash() const {
            int64_t h = 0;
            h ^= std::hash<double>{}(detuning) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<double>{}(duration)  + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<double>{}(phase)     + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<double>{}(amplitude) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

}

#endif