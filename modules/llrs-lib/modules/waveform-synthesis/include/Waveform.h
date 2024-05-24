#ifndef WAVEFORM_H_
#define WAVEFORM_H_

#include "llrs-lib/Settings.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <tuple>
#include <vector>

namespace Synthesis {

/// WaveformParameter type representing the (alpha, nu, phi) tuple
using WP = std::tuple<double, double, double>;

struct WaveformFunc { // Pure Abstract Waveform Mod type class
    virtual ~Waveform();
    virtual double transition_func() = 0;
    virtual double transition_func_integral() = 0;
};

class TANH : public WaveformFunc {
    // some parameters
  public:
    TANH();
    double transition_func() override;
    double transition_func_integral() override;
};

class Spline : public WaveformFunc {
    // some parameters
  public:
    Spline();
    double transition_func() override;
    double transition_func_integral() override;
};

class Step : public WaveformFunc {
    // some parameters
  public:
    Step();
    double transition_func() override;
    double transition_func_integral() override;
};

class ERF : public WaveformFunc {
    // some parameters
  public:
    ERF();
    double transition_func() override;
    double transition_func_integral() override;
};

class Static : public WaveformFunc {
    // some parameters
  public:
    Static();
    double transition_func() override;
    double transition_func_integral() override;
};

// WAVEFORM CLASS ---------------------------------------

class Waveform {
    double t0;
    double duration;
    double sample_rate;
    std::shared_ptr<WaveformFunc> wfMod;

  public:
    Waveform(std::shared_ptr<WaveformFunc> &wfMod, double t0, double duration,
             double sample_rate);
    std::vector<double> discretize(size_t num_samples);
    virtual double wave_func(size_t index) = 0;
}

class Displacement : public Waveform {
    std::shared_ptr<WaveformFunc> freqMod;
    WF destParams;
    void init_phase_adj();

  public:
    Displacement(std::shared_ptr<WaveformFunc> &ampMod,
                 std::shared_ptr<WaveformFunc> &freqMod, WP srcParams,
                 WP destParams)
        : Waveform(ampMod, srcParams), freqMod(freqMod),
          destParams(destParams) {
        init_phase_adj()
    }
    double wave_func(size_t index) override;
};

class Extraction : public Waveform {
    bool is_reversed;
    // params
  public:
    Extraction(std::shared_ptr<WaveformFunc> &wfMod, WP params,
               double waveform_duration, bool is_reversed)
        : Waveform(wfMod, params, 0, waveform_duration),
          is_reversed(is_reversed) {}
    double wave_func(size_t index) override;
};

class Idle : public Waveform {
  public:
    Idle(std::shared_ptr<WaveformFunc> &wfMod, WP params,
         double waveform_duration)
        : Waveform(wfMod, params, 0, waveform_duration) {}
    double wave_func(size_t index) override;
}

} // namespace Synthesis

#endif
