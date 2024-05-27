#ifndef MAIN_WAVEFORM_H_
#define MAIN_WAVEFORM_H_

#include "llrs-lib/Settings.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <tuple>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace Synthesis {

void read_waveform_configs(std::string filepath);
double read_waveform_duration(std::string filepath);

/// WaveformParameter type representing the (alpha, nu, phi) tuple
using WP = std::tuple<double, double, double>;

class TransitionFunc { // Pure Abstract Transition Mod type class
 protected:
    double duration;
	public:
    TransitionFunc(double duration) : duration(duration) {}
    ~TransitionFunc() {}
    virtual double transition_func(double t, WP params1, WP params2) = 0;
};

class StaticFunc {
	public:
    ~StaticFunc() {}
    virtual double static_func(double t, WP params) = 0;
};

class TANH : public TransitionFunc {
    double vmax;

  public:
    TANH(double duration, double vmax) : TransitionFunc(duration), vmax(vmax) {}
    ~TANH() {}
    double transition_func(double t, WP params1, WP params2) override;
};

class Spline : public TransitionFunc {

  public:
    Spline(double duration) : TransitionFunc(duration) {}
    ~Spline() {}
    double transition_func(double t, WP params1, WP params2) override;
};

class ERF : public TransitionFunc {
    double vmax;

  public:
    ERF(double duration, double vmax) : TransitionFunc(duration), vmax(vmax) {}
    ~ERF() {}
    double transition_func(double t, WP params1, WP params2) override;
};

class Sin : public StaticFunc {
	public:
    Sin() {}
    ~Sin() {}
    double static_func(double t, WP params) override;
};

// -------------- WAVEFORM CLASS -------------------

class Waveform {
  protected:
    double t0;
    double duration;
    double sample_rate;
    static std::unique_ptr<Synthesis::TransitionFunc> transMod;
    static std::unique_ptr<Synthesis::StaticFunc> staticMod;
    virtual double wave_func(double time) = 0;

  public:
    Waveform(double t0, double T) : t0(t0), duration(T) {}
    std::vector<double> discretize(size_t sample_rate);

    static void
    set_transition_function(std::unique_ptr<TransitionFunc> &&trans_func) {
        transMod = std::move(trans_func);
    }
    static void
    set_static_function(std::unique_ptr<StaticFunc> &&static_func) {
        staticMod = std::move(static_func);
    }
};

class Displacement : public Waveform {
  protected:
    WP srcParams, destParams;
    double wave_func(double time) override;

  public:
    Displacement(double duration, WP srcParams, WP destParams)
        : Waveform(0, duration), srcParams(srcParams), destParams(destParams) {}
};

class Extraction : public Waveform {
  protected:
    WP params;
    double wave_func(double time) override;

  public:
    Extraction(double duration, WP params)
        : Waveform(0, duration), params(params) {}
};

class Implantation : public Waveform {
  protected:
    WP params;
    double wave_func(double time) override;

  public:
    Implantation(double duration, WP params)
        : Waveform(0, duration), params(params) {}
};

class Idle : public Waveform {
  protected:
    WP params;
    double wave_func(double time) override;

  public:
    Idle(double duration, WP params) : Waveform(0, duration), params(params) {}
};

} // namespace Synthesis

#endif
