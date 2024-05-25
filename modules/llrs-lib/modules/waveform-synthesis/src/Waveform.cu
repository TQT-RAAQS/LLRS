#include <Waveform.h>

std::vector<double> Synthesis::Waveform::discretize(size_t sample_rate) {

    std::vector<double> data;
    double time;
    for (int i = 0; i < duration * sample_rate; i++) {
        time = i / sample_rate;
        data.push_back(wave_func(time));
    }
    return data;
}

double Synthesis::Extraction::wave_func(double time) {
    std::make_tuple<double, double, double> srcTuples(0, std::get<1>(params),
                                                      std::get<2>(params));

    return Synthesis::Waveform::transMod->transition_func(t, srcTuples, params);
}

double Synthesis::Implantation::wave_func(double time) {
    std::make_tuple<double, double, double> destParams(0, std::get<1>(params),
                                                       std::get<2>(params));

    return Synthesis::Waveform::transMod->transition_func(t, params,
                                                          destParams);
}

double Synthesis::Idle::wave_func(std::size_t index) {
    return Synthesis::Waveform::staticMod->static_func(t, params);
}

double Synthesis::Displacement::wave_func(std::size_t index) {
    return Synthesis::Waveform::transMod->transition_func(t, params,
                                                          destParams);
}

// ------------------ WAVEFORM FUNCTIONS ----------------

double Sin::transition_func(double t, WP params) {
    double alpha, nu, phi;
    std::tie(alpha, nu, phi) = params;

    return alpha * sin(2 * M_PI * nu * t + phi);
}

double ERF::transition_func(double t, WP params1, WP params2) {
    double alpha1, nu1, phi1;
    double alpha2, nu2, phi2;
    std::tie(alpha0, nu0, phi0) = params1;
    std::tie(alpha1, nu1, phi1) = params2;

    double alpha_mean = (alpha1 + alpha2) / 2;
    double dalpha = alpha2 - alpha_1;
    double alpha = alpha_mean + dalpha / 2 *
                                    erf(sqrt(M_PI) * vmax * (t - duration / 2);)

                                        double nu_mean = (nu1 + nu2) / 2;
    double dnu = nu2 - nu1;
    double phi_tilde =
        nu_mean * t +
        dnu / (2 * sqrt(M_PI) * vmax) *
            (sqrt(M_PI) * vmax * (t - duration / 2) *
                 erf(sqrt(M_PI) * vmax * (t - duration / 2)) +
             exp(-pow(sqrt(M_PI) * vmax * (t - duration / 2), 2)) / sqrt(M_PI) -
             sqrt(M_PI) * vmax * duration / 2 *
                 erf(sqrt(M_PI) * vmax * duration / 2) -
             exp(-pow(sqrt(M_PI) * vmax * (duration / 2), 2)) / sqrt(M_PI))

                double dphi = fmod(phi2 - phi1, 2 * M_PI);
    dphi = dphi - (abs(dphi) > M_PI) * (2 * (dphi > 0) - 1) * 2 * M_PI;

    return alpha * sin(phi1 + 2 * M_PI * phi_tilde + dphi * t / duration);
}

double TANH::transition_func(double t, WP params1, WP params2) {
    double alpha1, nu1, phi1;
    double alpha2, nu2, phi2;
    std::tie(alpha0, nu0, phi0) = params1;
    std::tie(alpha1, nu1, phi1) = params2;

    double alpha_mean = (alpha1 + alpha2) / 2;
    double dalpha = alpha2 - alpha_1;
    double alpha = alpha_mean + dalpha / 2 *
                                    tanh(2 * vmax * (t - duration / 2);)

                                        double nu_mean = (nu1 + nu2) / 2;
    double dnu = nu2 - nu1;
    double phi_tilde =
        nu_mean * t +
        dnu / (4 * vmax) *
            (log(cosh(2 * vmax * (t - T / 2))) - log(cosh(vmax * T)))

                double dphi = fmod(phi2 - phi1, 2 * M_PI);
    dphi = dphi - (abs(dphi) > M_PI) * (2 * (dphi > 0) - 1) * 2 * M_PI;

    return alpha * sin(phi1 + 2 * M_PI * phi_tilde + dphi * t / duration);
}

double Spline::transition_func(double t, WP params1, WP params2) {
    double alpha1, nu1, phi1;
    double alpha2, nu2, phi2;
    std::tie(alpha0, nu0, phi0) = params1;
    std::tie(alpha1, nu1, phi1) = params2;

    double a1 = -2 * (alpha2 - alpha1) * pow(t / duration, 3);
    double b1 = 3 * (alpha2 - alpha1) * pow(t / duration, 2);
    double alpha = a1 + b1 + alpha1;

    double a2 = -(nu2 - nu1) / (2 * pow(T, 3)) * pow(t, 4);
    double b2 = (nu2 - nu1) / (pow(T, 2)) * pow(t, 3);
    double phi_tilde = a2 + b2 + nu1 * t;

    double dphi = fmod(phi2 - phi1 - phi_tilde, 2 * M_PI);
    dphi = dphi - (abs(dphi) > M_PI) * (2 * (dphi > 0) - 1) * 2 * M_PI;

    return alpha * sin(phi1 + 2 * M_PI * phi_tilde + dphi * t / duration);
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

	double waveform_duration = node["waveform_duration"].as<double>();

    // Transition Mod
    std::string transition_type = node["transition_type"].as<std::string>();
    if (transition_type == "TANH") {
        Waveform::set_transition_function(
            std::make_unique<TANH>(waveform_duration, node["transition_type"]["vmax"].as<double>()));
    } else if (transition_type == "Spline") {
        Waveform::set_transition_function(
            std::make_unique<Spline>(waveform_duration));
    } else if (transition_type == "Step") {
        Waveform::set_transition_function(
            std::make_unique<Step>(waveform_duration));
    } else if (transition_type == "ERF") {
        Waveform::set_transition_function(
            std::make_unique<ERF>(waveform_duration, node["transition_type"]["vmax"].as<double>()));   } else {
        throw std::invalid_argument(
            "Transition modulation type not supported.");
    }

    // Static Mod
    std::string static_type == node["static_type"].as<std::string>();
    Waveform::staticMod = std::make_unique<Spline>(node["static_type"]);
    if (transition_type == "sin") {
        Waveform::set_static_function(
            std::make_unique<Sin>());
    } else {
        throw std::invalid_argument("Static modulation type not supported.");
    }
}

double read_waveform_duration(std::string filepath) {
    /// Open file
    try {
        YAML::Node node = YAML::LoadFile(filename);
    } catch (const YAML::BadFile &e) {
        std::cerr << "Error loading YAML file (Waveform Config)." << std::endl;
        std::cerr << "ERROR: " << e.what() << std::endl;
        return;
    }

	return node["waveform_duration"].as<double>();
}
