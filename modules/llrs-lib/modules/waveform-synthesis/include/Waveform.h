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
enum Waveform::Type { DISPLACEMENT, EXTRACTION, IMPLANTATION };

class Waveform { // Pure Abstract Waveform class
  protected:
    WP params;
    double t0;
    double duration;
    double sample_rate;

  public:
    Waveform(WP params, double t0, double T, double sample_rate);
    virtual ~Waveform();

    std::vector<double> discretize(Waveform::Type type, size_t num_samples);
    virtual double wave_func(Waveform::Type type, size_t index) = 0;
};

class TANH : public Waveform {
    // some parameters
    transition_func();
    transition_func_integral() public
        : TANH(WP params, double t0, double T, double sample_rate);
    double wave_func(Waveform::Type type, size_t index) override;
};

class Spline : public Waveform {
    // some parameters
    transition_func();
    transition_func_integral() public
        : Spline(WP params, double t0, double T, double sample_rate);
    double wave_func(Waveform::Type type, size_t index) override;
};

class Step : public Waveform {
    // some parameters
    transition_func();
    transition_func_integral() public
        : Step(WP params, double t0, double T, double sample_rate);
    double wave_func(Waveform::Type type, size_t index) override;
};

class ERF : public Waveform {
    // some parameters
    transition_func();
    transition_func_integral() public
        : ERF(WP params, double t0, double T, double sample_rate);
    double wave_func(Waveform::Type type, size_t index) override;
};

class Static : public Waveform {
    // some parameters
    transition_func();
    transition_func_integral() public
        : Static(WP params, double t0, double T, double sample_rate);
    double wave_func(Waveform::Type type, size_t index) override;
};

} // namespace Synthesis

#endif
