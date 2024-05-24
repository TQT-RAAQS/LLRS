#ifndef WAVEFORM_H_
#define WAVEFORM_H_

#include "llrs-lib/Settings.h"
#include <yaml-cpp/yaml.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <tuple>
#include <vector>

namespace Synthesis {

void read_waveform_configs(std::string filepath);

/// WaveformParameter type representing the (alpha, nu, phi) tuple
using WP = std::tuple<double, double, double>;

struct TransitionFunc { // Pure Abstract Transition Mod type class
    virtual ~TransitionFunc();
    virtual double transition_func() = 0;
    virtual double transition_func_integral() = 0;
};

struct StaticFunc {
	virtual ~StaticFunc();
	virtual double static_func() = 0;
};

class TANH : public TransitionFunc {
    // some parameters
  public:
    TANH();
    double transition_func() override;
    double transition_func_integral() override;
};

class Spline : public TransitionFunc {
    // some parameters
  public:
    Spline();
    double transition_func() override;
    double transition_func_integral() override;
};

class Step : public TransitionFunc {
    // some parameters
  public:
    Step();
    double transition_func() override;
    double transition_func_integral() override;
};

class ERF : public TransitionFunc {
    // some parameters
  public:
    ERF();
    double transition_func() override;
    double transition_func_integral() override;
};

class Sin : public StaticFunc {
    // some parameters
  public:
	Sin();
    double transition_func() override;
    double transition_func_integral() override;
};

// WAVEFORM CLASS ---------------------------------------

class Waveform {
	WP params;
    double t0;
    double duration;
    double sample_rate;
    static std::unique_ptr<TransitionFunc> transMod;
	static std::unique_ptr<StaticFunc> staticMod;
    virtual double wave_func(size_t index) = 0;

  public:
    Waveform(double t0, double duration);
    std::vector<double> discretize(size_t num_samples, double sample_rate);
}

class Displacement : public Waveform {
    WF destParams;
    void init_phase_adj();

  public:
    Displacement(WP srcParams, WP destParams, double waveform_duration)
        : Waveform(srcParams, 0, waveform_duration), destParams(destParams) {
        init_phase_adj()
    }
};

class Extraction : public Waveform {
    bool is_reversed;
    // params
    double wave_func(size_t index) override;
  public:
    Extraction(WP params, double waveform_duration, bool is_reversed)
        : Waveform(params, 0, waveform_duration),
          is_reversed(is_reversed) {}
};

class Idle : public Waveform {
    double wave_func(size_t index) override;
  public:
    Idle(WP params, double waveform_duration)
        : Waveform(params, 0, waveform_duration) {}
}

} // namespace Synthesis

#endif
