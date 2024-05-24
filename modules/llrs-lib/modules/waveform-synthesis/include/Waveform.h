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
    double duration;

    TransitionFunc(double duration) : duration(duration) {}
    virtual ~TransitionFunc();
    virtual double transition_func(double t, WP params1, WP params2) = 0;
};

struct StaticFunc {
  double duration;

	virtual ~StaticFunc();
	virtual double static_func(double t, WP params) = 0;
};

class TANH : public TransitionFunc {
    double vmax;

  public:
    TANH(double duration, double vmax) : TransitionFunc(duration), vmax(vmax) {}
    double transition_func(double t, WP params1, WP params2) override;
};

class Spline : public TransitionFunc {
    
  public:
    Spline(double duration) : TransitionFunc(duration) {}
    double transition_func(double t, WP params1, WP params2) override;
};

class ERF : public TransitionFunc {
  double vmax;
  
  public:
    ERF(double duration, double vmax) : TransitionFunc(duration), vmax(vmax) {}
    double transition_func(double t, WP params1, WP params2) override;
};

class Sin : public StaticFunc {
    
  public:
    Sin();
    double transition_func(double t, WP params) override;
};

// --------------------------------------- WAVEFORM CLASS ---------------------------------------

class Waveform {
	WP params;
    double t0;
    double duration;
    double sample_rate;
    static std::unique_ptr<TransitionFunc> transMod;
	  static std::unique_ptr<StaticFunc> staticMod;
    virtual double wave_func(double time) = 0;

  public:
    Waveform(double t0, double T) : t0(t0), duration(T) {}
    std::vector<double> discretize(size_t num_samples, double sample_rate);
    
    static void set_transition_function(std::unqiue_ptr<TransitionFunc> && trans_func) {transMod = trans_func;}
    static void set_static_function(std::unqiue_ptr<TransitionFunc> && static_func) {staticMod = static_func;}
}

class Displacement : public Waveform {
    WF srcParams, destParams;
    void init_phase_adj();
    double wave_func(double time) override;

  public:
    Displacement(double duration, WP srcParams, WP destParams)
        : Waveform(0, duration), srcParams(srcParams), destParams(destParams) {
        init_phase_adj()
    }
};

class Extraction : public Waveform {
    WF params;
    double wave_func(double time) override;

  public:
    Extraction(double duration, WP params)
        : Waveform(0, duration), params(params) {}
};

class Implantation : public Waveform {
    WF params;
    double wave_func(double time) override;

  public:
    Implantation(double duration, WP params)
        : Waveform(0, duration), params(params) {}
};

class Idle : public Waveform {
    WF params;
    double wave_func(double time) override;

  public:
    Idle(double duration, WP params)
        : Waveform(0, duration), params(params) {}
}

} // namespace Synthesis

#endif
