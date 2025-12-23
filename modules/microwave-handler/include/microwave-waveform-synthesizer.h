#ifndef _MICROWAVE_WAVEFORM_SYNTHESIZER_H_
#define _MICROWAVE_WAVEFORM_SYNTHESIZER_H_

#include "microwave-waveforms.h"
#include "llrs-lib/PreProc.h"
#include "configs-translator.h"
#include "awg.hpp"
#include <vector>
#include <tuple>

namespace MicrowaveHandler {

    class MicrowaveWaveformSynthesizer {

        ConfigsTranslator& config_translator = ConfigsTranslator::instance();

        bool flag_fast_interleaving = false;

        double dphi, vI_dc, vQ_dc;
        double digital_offset_time = 0;

        std::vector<short> v_I;
        std::vector<short> v_Q;
        std::vector<int8> digital_trigger;

        void generate_square_pulse(
            short* buffer,
            size_t sample_count,
            AWG& awg,
            const MicrowaveHandler::SquarePulse* p,
            double t,
            double t_initial_pause);

        void generate_square_pulse_fast(
            short* buffer,
            size_t sample_count,
            AWG& awg,
            const MicrowaveHandler::SquarePulse* p,
            double t,
            double t_initial_pause);

    public:

        MicrowaveWaveformSynthesizer();

        void reload(bool flag_translate = true);

        void generate_pulse(short* buffer,
                        size_t sample_count,
                        AWG& awg,
                        const MicrowaveHandler::Waveform& iqmixer_waveform,
                        double t,
                        double t_initial_pause);

        void set_digital_offset_time(double digital_offset_time) { this->digital_offset_time = digital_offset_time; }
        void set_fast_interleaving_flag(bool flag) { this->flag_fast_interleaving = flag; }

    };

}

#endif