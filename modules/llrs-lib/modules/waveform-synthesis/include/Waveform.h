#ifndef _WAVEFORM_H_
#define _WAVEFORM_H_

#include "LLRS-lib/Settings.h"
#include <tuple>
#include <vector>
#include <stdio.h>
#include <cmath>
#include <cassert>
#include <iostream>


namespace Synthesis {

    /// WaveformParameter type representing the (alpha, nu, phi) tuple
    using WP = std::tuple<double, double, double>;


    /// Modulation Enum representing the Modulation Type
    enum Modulation {
        STEP,
        TANH,
        ERF
    };


    /**
    * @brief Calculates a modulation function based on the specified modulation type.
    * @param mod_type  => modulation type
    * @param time 
    * @param relative_slope
    * @param duration
    */ 
    double transition_func(Modulation mod_type, double time, double relative_slope, double duration);


    /**
    * @brief Calculates the definite integral of a modulation function over the range [a,b] based on the modulation type
    * @param integrand
    * @param a => lower bound of the integration range
    * @param b => upper bound of the integration range
    */
    double transition_func_integral(Modulation integrand, double b, double a);


    /// Static Waveform Base Class
    class Waveform {
        WP _params;

    protected:    
        double _t0;
        double _duration;
 
    public:
        /**
         * @brief Constructor for Waveform class
         * @param WP => the Tuple of waveform params
         * @param t0 => start time
         * @param T  => Waveform duration
        */
        Waveform(WP params, double t0 = 0, double T = WAVEFORM_DUR);

        /// Getters/Setters
        WP get_params()                 { return _params; }
        double get_start_time()         { return _t0; }
        void set_start_time(double t0)  { this->_t0 = t0; }


        /**
         * @brief Generates a waveform by evaluating a given wavefunciton for a specified number of samples at a given sample rate
         *  @param num_samples
         *  @param sample_rate
         */ 
        std::vector<double> discretize(std::size_t num_samples = WAVEFORM_LEN, double sample_rate = AWG_SAMPLE_RATE);


        /**
        * @brief Generates a waveform sample at a specified index using parameters 
        * @param index  => index to generate the waveform at
        * @param sample_rate => sample rate of the waveform
        */
        virtual double wave_func(std::size_t index, double sample_rate);

    };


    class Extraction: public Waveform {
        bool       _is_reversed;
        Modulation _amp_mod;
 
    public:
        /**
         * @brief Constructor for Extraction class
         * @param params  => to construct the base class
         * @param is_reversed 
         * @param amp_mod  => Modulation type
         */
        Extraction(WP params, bool is_reversed, Modulation amp_mod = TANH): Waveform(params), _is_reversed(is_reversed), _amp_mod(amp_mod) {}


        /**
        * @brief generates an extraction waveform sample at a specified index using parameters obtained from get_params(), and includes amplitude modulation
        * @param index  => index to generate the waveform at
        * @param sample_rate => sample rate of the waveform
        * @return Waveform
        */
        double wave_func(std::size_t index, double sample_rate) override;
   };


    class Displacement: public Waveform {
        WP         _dest;
        Modulation _amp_mod;
        Modulation _freq_mod;
        double     _tau;
        double     _v_max;
        double     _delta;
    
    public:
        /**
         * @brief Constructor for Displacement class
         * @param params => To construct base class
         * @param dest 
         * @param amp_mod
         * @param freq_mod
         * @param freq_slope 
        */
        Displacement(WP params, WP dest, Modulation amp_mod = TANH, Modulation freq_mod = ERF, double freq_slope = DISPLACEMENT_FREQUENCY_RELATIVE_SLOPE);

        /**
        * @brief generates a displacement waveform sample at a specified index using parameters obtained from get_params(), and includes 
        * amplitude modulation and frequency modulation
        * @param index  => index to generate the waveform at
        * @param sample_rate => sample rate of the waveform
        */
        double wave_func(std::size_t index, double sample_rate) override;

        /**
        * @brief initializes the phase adjustment of a displacement waveform
        */
        void   init_phase_adj();
   };

}


#endif
