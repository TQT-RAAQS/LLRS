#include "microwave-waveform-synthesizer.h"

using namespace MicrowaveHandler;

#define MW_MAX_ANALOG_VALUE 0x7fff
#define ALPHA_MAX 0.083

MicrowaveWaveformSynthesizer::MicrowaveWaveformSynthesizer() {
    this->reload();
}

void MicrowaveWaveformSynthesizer::reload(bool flag_translate) {
    this->reload_iqmixer_parameters(flag_translate);
    this->ramsey_stabilizer_60hz_model.reload_parameters();
}

void MicrowaveWaveformSynthesizer::reload_iqmixer_parameters(bool flag_translate) {
    if (flag_translate) {
        this->config_translator.translate_iqmixer();
    }
    std::ifstream infile(IQMIXER_TRANSLATION_FILE, std::ios::binary);
    infile.read(reinterpret_cast<char*>(&this->dphi), sizeof(double));
    infile.read(reinterpret_cast<char*>(&this->vI_dc), sizeof(double));
    infile.read(reinterpret_cast<char*>(&this->vQ_dc), sizeof(double));
    infile.close();
}

void MicrowaveWaveformSynthesizer::generate_pulse(
    short* buffer,
    size_t sample_count,
    AWG& awg,
    const MicrowaveHandler::Waveform& iqmixer_waveform,
    double t,
    double t_initial_pause) {
        
        if (auto *p = boost::get<SquarePulse>(&iqmixer_waveform)) {
            if (this->flag_fast_interleaving) {
                this->generate_square_pulse_fast(buffer, sample_count, awg, p, t, t_initial_pause);
             } else{
                this->generate_square_pulse(buffer, sample_count, awg, p, t, t_initial_pause);
             }
            return;
        }
        if (auto *p = boost::get<Square60Pulse>(&iqmixer_waveform)) {
            this->generate_square60_pulse(buffer, sample_count, awg, p, t, t_initial_pause);
            return;
        }

        throw std::runtime_error("Waveform type not detected: " + std::to_string(iqmixer_waveform.which()));
}

void MicrowaveWaveformSynthesizer::generate_square_pulse(
    short* buffer,
    size_t sample_count,
    AWG& awg,
    const SquarePulse* p,
    double t0,
    double t_initial_pause)
{
    v_I.resize(sample_count);
    v_Q.resize(sample_count);
    digital_trigger.resize(sample_count);

    double dt = 1.0 / awg.get_sample_rate();

    // Normalization factors
    if (p->amplitude > ALPHA_MAX) {
        throw std::runtime_error(
            "SquarePulse amplitude exceeds maximum allowed value of " +
            std::to_string(ALPHA_MAX) + ": " + std::to_string(p->amplitude)
        );
    }

    double a0 = awg.get_amplitude(0) * 1e-3;
    double a1 = awg.get_amplitude(1) * 1e-3;
    double alpha_I = p->amplitude / a0;
    double alpha_Q = p->amplitude / a1;
    double alpha_I_dc = this->vI_dc / a0;
    double alpha_Q_dc = this->vQ_dc / a1;

    int pause_samples = static_cast<int>(round(t_initial_pause / dt));
    int pulse_samples = static_cast<int>(round(p->duration / dt));
    int offset_samples = static_cast<int>(round(abs(digital_offset_time) / dt));
    int expected_samples = pause_samples + pulse_samples + offset_samples; // Minimum expected samples

    if (sample_count < static_cast<size_t>(expected_samples)) {
        std::cout << sample_count << " " << expected_samples << std::endl;
        throw std::runtime_error(
            "Mismatch between sample_count and expected_samples: " +
            std::to_string(sample_count) + " < " + std::to_string(expected_samples)
        );
    }

    // Zero initial pause
    std::fill(v_I.begin(), v_I.begin() + pause_samples, 0);
    std::fill(v_Q.begin(), v_Q.begin() + pause_samples, 0);
    std::fill(digital_trigger.begin(), digital_trigger.begin() + pause_samples, 0);
    
    // Generate pulse
    auto t_init = t0 + pause_samples * dt;
    #pragma omp simd
    for (int i = 0; i < pulse_samples + offset_samples; ++i) {
        double t_rel = i * dt;  // relative to pulse start
        auto tnow = t_init + t_rel;
        auto freq = p->frequency;
        double mask = (t_rel >= -digital_offset_time && t_rel <= p->duration - digital_offset_time);

        v_I[pause_samples + i] = static_cast<short>(MW_MAX_ANALOG_VALUE * mask * (alpha_I * sin(2 * M_PI * freq * tnow + p->phase) + alpha_I_dc));
        v_Q[pause_samples + i] = static_cast<short>(MW_MAX_ANALOG_VALUE * mask * (alpha_Q * sin(2 * M_PI * freq * tnow + p->phase + dphi) + alpha_Q_dc));

        digital_trigger[pause_samples + i] = (t_rel >= digital_offset_time && t_rel <= p->duration + digital_offset_time) ? 1 : 0;
    }

    // Zero remaining samples if any
    if (static_cast<size_t>(expected_samples) < sample_count) {
        std::fill(v_I.begin() + expected_samples, v_I.end(), 0);
        std::fill(v_Q.begin() + expected_samples, v_Q.end(), 0);
        std::fill(digital_trigger.begin() + expected_samples, digital_trigger.end(), 0);
    }

    awg.interleave_data(buffer, {v_I, v_Q}, {digital_trigger});
}

void MicrowaveWaveformSynthesizer::generate_square60_pulse(
    short* buffer,
    size_t sample_count,
    AWG& awg,
    const Square60Pulse* p,
    double t0,
    double t_initial_pause)
{
    v_I.resize(sample_count);
    v_Q.resize(sample_count);
    digital_trigger.resize(sample_count);

    double dt = 1.0 / awg.get_sample_rate();

    // Normalization factors
    if (p->amplitude > ALPHA_MAX) {
        throw std::runtime_error(
            "Square60Pulse amplitude exceeds maximum allowed value of " +
            std::to_string(ALPHA_MAX) + ": " + std::to_string(p->amplitude)
        );
    }

    double a0 = awg.get_amplitude(0) * 1e-3;
    double a1 = awg.get_amplitude(1) * 1e-3;
    double alpha_I = p->amplitude / a0;
    double alpha_Q = p->amplitude / a1;
    double alpha_I_dc = this->vI_dc / a0;
    double alpha_Q_dc = this->vQ_dc / a1;

    int pause_samples = static_cast<int>(round(t_initial_pause / dt));
    int pulse_samples = static_cast<int>(round(p->duration / dt));
    int offset_samples = static_cast<int>(round(abs(digital_offset_time) / dt));
    int expected_samples = pause_samples + pulse_samples + offset_samples; // Minimum expected samples

    if (sample_count < static_cast<size_t>(expected_samples)) {
        std::cout << sample_count << " " << expected_samples << std::endl;
        throw std::runtime_error(
            "Mismatch between sample_count and expected_samples: " +
            std::to_string(sample_count) + " < " + std::to_string(expected_samples)
        );
    }

    // Zero initial pause
    std::fill(v_I.begin(), v_I.begin() + pause_samples, 0);
    std::fill(v_Q.begin(), v_Q.begin() + pause_samples, 0);
    std::fill(digital_trigger.begin(), digital_trigger.begin() + pause_samples, 0);
    
    // Generate pulse
    auto t_init = t0 + pause_samples * dt;
    #pragma omp simd
    for (int i = 0; i < pulse_samples + offset_samples; ++i) {
        double t_rel = i * dt;  // relative to pulse start
        auto tnow = t_init + t_rel;
        auto freq = p->frequency;
        double mask = (t_rel >= -digital_offset_time && t_rel <= p->duration - digital_offset_time);
        double phase_correction = this->ramsey_stabilizer_60hz_model.get_phase_correction(tnow);

        v_I[pause_samples + i] = static_cast<short>(MW_MAX_ANALOG_VALUE * mask * (alpha_I * sin(2 * M_PI * freq * tnow + p->phase + phase_correction) + alpha_I_dc));
        v_Q[pause_samples + i] = static_cast<short>(MW_MAX_ANALOG_VALUE * mask * (alpha_Q * sin(2 * M_PI * freq * tnow + p->phase + dphi + phase_correction) + alpha_Q_dc));

        digital_trigger[pause_samples + i] = (t_rel >= digital_offset_time && t_rel <= p->duration + digital_offset_time) ? 1 : 0;
    }

    // Zero remaining samples if any
    if (static_cast<size_t>(expected_samples) < sample_count) {
        std::fill(v_I.begin() + expected_samples, v_I.end(), 0);
        std::fill(v_Q.begin() + expected_samples, v_Q.end(), 0);
        std::fill(digital_trigger.begin() + expected_samples, digital_trigger.end(), 0);
    }

    awg.interleave_data(buffer, {v_I, v_Q}, {digital_trigger});
}

void MicrowaveWaveformSynthesizer::generate_square_pulse_fast(
    short* buffer,
    size_t sample_count,
    AWG& awg,
    const SquarePulse* p,
    double t0,
    double t_initial_pause)
{
    if (awg.get_sync_digital_out_num_channels() != 1) {
        throw std::runtime_error(
            "Fast interleaving method requires exactly one digital sync output channel."
        );
    }
    if (awg.get_num_channels() != 2) {
        throw std::runtime_error(
            "Fast interleaving requires two configured awg channels."
        );
    }
    auto digital_channel = (awg.get_sync_digital_out_configs()[0].channel == awg.get_channels()[0] ? 0 : 1);
    auto digital_bit = awg.get_sync_digital_out_configs()[0].bit;
    auto analog_bit = 16 - digital_bit;

    double dt = 1.0 / awg.get_sample_rate();

    // Normalization factors
    if (p->amplitude > ALPHA_MAX) {
        throw std::runtime_error(
            "SquarePulse amplitude exceeds maximum allowed value of " +
            std::to_string(ALPHA_MAX) + ": " + std::to_string(p->amplitude)
        );
    }

    double a0 = awg.get_amplitude(0) * 1e-3;
    double a1 = awg.get_amplitude(1) * 1e-3;
    double alpha_I = p->amplitude / a0;
    double alpha_Q = p->amplitude / a1;
    double alpha_I_dc = this->vI_dc / a0;
    double alpha_Q_dc = this->vQ_dc / a1;

    int pause_samples = static_cast<int>(round(t_initial_pause / dt));
    int pulse_samples = static_cast<int>(round(p->duration / dt));
    int offset_samples = static_cast<int>(round(abs(digital_offset_time) / dt));
    int expected_samples = pause_samples + pulse_samples + offset_samples;

    if (sample_count < static_cast<size_t>(expected_samples)) {
        std::cout << sample_count << " " << expected_samples << std::endl;
        throw std::runtime_error(
            "Mismatch between sample_count and expected_samples: " +
            std::to_string(sample_count) + " < " + std::to_string(expected_samples)
        );
    }

    // Zero initial pause
    std::fill(buffer, buffer + 2 * pause_samples, 0);

    // Generate pulse
    auto t_init = t0 + pause_samples * dt;
    #pragma omp simd
    for (int i = 0; i < pulse_samples + offset_samples; ++i) {
        auto t_rel = i * dt;  // relative to pulse start
        auto tnow = t_init + t_rel;
        auto freq = p->frequency;
        auto mask = static_cast<double>(t_rel >= -digital_offset_time && t_rel <= p->duration - digital_offset_time);

        auto vI = static_cast<short>(MW_MAX_ANALOG_VALUE * mask * (alpha_I * sin(2 * M_PI * freq * tnow + p->phase) + alpha_I_dc));
        auto vQ = static_cast<short>(MW_MAX_ANALOG_VALUE * mask * (alpha_Q * sin(2 * M_PI * freq * tnow + p->phase + dphi) + alpha_Q_dc));
        auto vD = static_cast<int16>( (t_rel >= digital_offset_time && t_rel <= p->duration + digital_offset_time) ? 1 : 0 );

        buffer[2 * (pause_samples + i)] = (digital_channel == 0 ? static_cast<short>( static_cast<uint16>(vI) >> analog_bit | (vD << digital_bit) ) : vI);
        buffer[2 * (pause_samples + i) + 1] = (digital_channel == 1 ? static_cast<short>( static_cast<uint16>(vQ) >> analog_bit | (vD << digital_bit) ) : vQ);
    }

    // Zero remaining samples if any
    if (static_cast<size_t>(expected_samples) < sample_count) {
        std::fill(buffer + 2 * expected_samples, buffer + 2 * sample_count, 0);
    }
}

