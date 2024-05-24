#include "Waveform.h"

/// Helper Functions
double Synthesis::transition_func(Modulation mod_type, double time,
                                  double relative_slope, double duration) {
    double eval =
        relative_slope * (time - duration / 2); // caluclate modulation function
    // switch based on modulation type
    switch (mod_type) {
    case STEP:
        return 2 * eval / sqrt(1 + 4 * eval * eval);
    case TANH:
        return tanh(2 * eval);
    case ERF:
        return erf(sqrt(M_PI) * eval);
    default:
        throw std::invalid_argument(
            "Synthesis::transition_func() -> Function not supported.");
    }
}

double Synthesis::transition_func_integral(Modulation integrand, double b,
                                           double a) {
    double upper = 0;
    double lower = 0;

    /// for each modulation type, the value of the antiderivative is calculated
    /// at the upper and lower bound of the integral
    switch (integrand) {
    case STEP:
        upper = sqrt(4 * b * b + 1) / 2;
        lower = sqrt(4 * a * a + 1) / 2;
        break;

    case TANH:
        upper = log(cosh(2 * b)) / 2;
        lower = log(cosh(2 * a)) / 2;
        break;

    case ERF:
        upper = b * erf(sqrt(M_PI) * b) + exp(-M_PI * (b * b)) / M_PI;
        lower = a * erf(sqrt(M_PI) * a) + exp(-M_PI * (a * a)) / M_PI;
        break;

    default: /// throw an exception if the modulation type is unsupported
        throw std::invalid_argument(
            "ERROR: Synthesis::transition_func_integral() -> Function not "
            "supported.");
    }

    return upper - lower;
}

/// Waveform Class
Synthesis::Waveform::Waveform(std::shared_ptr<WaveformFunc> &wfMod, WP params,
                              double t0, double T)
    : wfFunc(wfFunc), params(params), t0(t0), duration(T) {
    double alpha, nu, phi;
    std::tie(alpha, nu, phi) = params;
    if (alpha > 1 || alpha < 0)
        throw std::invalid_argument("Invalid Amplitude Values.");
    if (nu <= 0)
        throw std::invalid_argument("Invalid Frequency Values.");
    if (phi >= 2 * M_PI || phi <= -2 * M_PI)
        throw std::invalid_argument("Invalid Phase Values.");
};

std::vector<double> Synthesis::Waveform::discretize(size_t num_samples) {

    std::vector<double> data;
    for (int i = 0; i < num_samples; i++) {
        data.push_back(wave_func(i, sample_rate));
    }
    return data;
}

// Extraction
double Synthesis::Extraction::wave_func(size_t index) {
    double alpha, nu, phi;
    std::tie(alpha, nu, phi) = params;

    double mod_val = transition_func(this->_amp_mod, index / sample_rate,
                                     DISPLACEMENT_RELATIVE_SLOPE, _duration);
    double mod_coef = (this->_is_reversed) ? -alpha / 2 : alpha / 2;
    double amp_chirp = mod_coef * mod_val + (alpha / 2);
    return amp_chirp * sin(2 * M_PI * nu * (_t0 + index / sample_rate) + phi);
}

// Idle
double Synthesis::Idle::wave_func(std::size_t index) {
    double alpha, nu, phi;
    std::tie(alpha, nu, phi) = params;

    /// The waveform is a sine wave with amplitude alpha, frequency nu, phase
    /// shift phi, and an optional time offset t0
    return alpha * sin(2 * M_PI * nu * (t0 + index / sample_rate) + phi);
}

// Displacement

double Synthesis::Displacement::wave_func(std::size_t index) {
    double alpha0, nu0, phi0;
    double alpha1, nu1, phi1;
    std::tie(alpha0, nu0, phi0) = params;
    std::tie(alpha1, nu1, phi1) = destParams;

    double alpha_mod = transition_func(this->_amp_mod, index / sample_rate,
                                       DISPLACEMENT_RELATIVE_SLOPE, _duration);
    double amp_chirp =
        ((alpha1 - alpha0) / 2) * alpha_mod + (alpha1 + alpha0) / 2;

    // Frequency modulation
    double nu_bar = (nu1 + nu0) / 2;
    double v_ratio = this->_v_max / abs(nu1 - nu0);
    double integ_a = v_ratio * (-_duration / 2 - _tau);
    double integ_b = v_ratio * ((index / sample_rate) - _duration / 2 - _tau);
    double nu_mod = transition_func_integral(this->_freq_mod, integ_b, integ_a);
    double freq_chirp = ((nu1 - nu0) / 2) * (nu_mod / v_ratio);

    return amp_chirp *
           sin(2 * M_PI * (freq_chirp + nu_bar * (index / sample_rate)) +
               phi0); // return waveform
}

void Synthesis::Displacement::init_phase_adj() {
    double alpha0, nu0, phi0;
    double alpha1, nu1, phi1;
    std::tie(alpha0, nu0, phi0) = params;
    std::tie(alpha1, nu1, phi1) = destParams;

    // Calculate phase adjustment
    double nu_bar = (nu1 + nu0) / 2;
    double delta = (phi1 - phi0) / (2 * M_PI);
    double candidate = nu_bar * _duration - delta;

    double tau_ceil = (candidate - ceil(candidate)) / (nu1 - nu0);
    double tau_floor = (candidate - floor(candidate)) / (nu1 - nu0);

    this->_tau = (abs(tau_ceil) < abs(tau_floor))
                     ? tau_ceil
                     : tau_floor; // stores the phase adjustment in the waveform
}


void read_waveform_configs(std::string filepath) {
    /// Open file
    try {
		YAML::Node node = YAML::LoadFile(filename);
    } catch (const YAML::BadFile &e) {
        std::cerr << "Error loading YAML file (Waveform Config)." << std::endl;
        std::cerr << "ERROR: " << e.what() << std::endl;
		return;
    }
	
	
	// Transition Mod
	std::string transition_type == node["transition_type"].as<std::string>();
	if (transition_type == "TANH") {
		Waveform::transMod = std::make_unique<TANH>(node["transition_type"]);	
	} else if (transition_type == "Spline") {
		Waveform::transMod = std::make_unique<Spline>(node["transition_type"]);	
	} else if (transition_type == "Step") {
		Waveform::transMod = std::make_unique<Step>(node["transition_type"]);	
	} else if (transition_type == "ERF") {
		Waveform::transMod = std::make_unique<ERF>(node["transition_type"]);	
	} else {
		throw std::invalid_argument("Transition modulation type not supported.");
	}

	// Static Mod
	std::string static_type == node["static_type"].as<std::string>();
	Waveform::staticMod = std::make_unique<Spline>(node["static_type"]);	
	if (transition_type == "sin") {
		Waveform::staticMod = std::make_unique<Sin>(node["static_type"]);	
	} else {
		throw std::invalid_argument("Static modulation type not supported.");
	}
			
}
