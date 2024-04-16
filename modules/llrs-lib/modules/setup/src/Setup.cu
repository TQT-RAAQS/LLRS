/**
 * @brief Parameter handling and waveform synthesis setup
 * @date Nov 2023
 */
#include "Setup.h"

/**
 * @brief reads file parameters and saves them into a vector of tuples
 * (optimized alpha, phi, nu)
 * @param file_name => name of file
 * @param params => vector of tuples for parameters, will be changed
 * @param limit => number of traps
 * @return status code
 */

int Setup::read_fparams(std::string file_name,
                        std::vector<Synthesis::WP> &params, const int limit) {

    std::fstream fin;
    fin.open(file_name);

    std::string line;
    getline(fin, line);

    int count = 0;
    while (!fin.eof() && count < limit) {
        count++;

        std::string param;
        std::vector<double> tmp_params;

        getline(fin, line);
        std::stringstream sline(line);

        while (getline(sline, param, ',')) {
            if (isnan(stod(param))) {
                fin.close();
                throw std::invalid_argument("ERROR: waveform_table->params: "
                                            "waveform parameter is null");
            }
            tmp_params.push_back(stod(param));
        }

        if (tmp_params.size() != PARAM_NUM) {
            fin.close();
            throw std::invalid_argument("ERROR: waveform_table->params: not "
                                        "enough waveform parameters");
        }

        params.push_back(
            std::make_tuple(tmp_params[0], tmp_params[1], tmp_params[2]));
    }

    fin.close();

    if (count < limit) {
        throw std::invalid_argument(
            "ERROR: waveform_table->params: not enough waveform parameters");
    }

    return LLRS_OK;
}

/**
 * @brief sets up waveform repository with given parameters, which is then
 * updated with waveform data and saved in binary
 * @param Nt_x => number of traps (smaller of the two dimensions)
 * @param Nt_y => number of traps (larger of the two dimensions)
 * @param sample_rate => waveform sampling rate
 * @param wf_duration => waveform duration
 * @param coef_x_fname => file name for file containing diffraction of AOD in
 * the X direction
 * @param coef_y_fname => file name for file containing diffraction of AOD in
 * the Y direction
 * @return status code of entire process
 */

int Setup::create_wf_repo(size_t Nt_x, size_t Nt_y, double sample_rate,
                          double wf_duration, int waveform_length,
                          int waveform_mask, int vpp, std::string coef_x_fname,
                          std::string coef_y_fname) {

    Synthesis::WaveformRepo wf_repo(Nt_x, Nt_y, sample_rate, wf_duration,
                                    waveform_length, waveform_mask, vpp);

    std::string x_path = COEF_X_PATH(coef_x_fname);
    std::vector<Synthesis::WP> params_x;
    if (FILE_EXISTS(x_path)) {
        read_fparams(x_path, params_x, Nt_x);
    } else {
        return LLRS_ERR;
    }

    std::string y_path = COEF_Y_PATH(coef_y_fname);
    std::vector<Synthesis::WP> params_y;
    if (FILE_EXISTS(y_path)) {
        read_fparams(y_path, params_y, Nt_y);
    } else {
        return LLRS_ERR;
    }

    wf_repo.generate_repository(params_x, params_y);
    std::string file_name = "repo." + std::to_string(Nt_x) + "_" +
                            std::to_string(Nt_y) + "_" +
                            std::to_string((int)std::round(sample_rate)) + "_" +
                            std::to_string((int)std::round(wf_duration * 1e6)) +
                            "_" + std::to_string(waveform_length) + "_" +
                            std::to_string(waveform_mask) + ".bin";
    std::string repo_fname = WF_REPO_PATH(file_name);
    INFO << "Caching repo as" << repo_fname << std::endl;

    return wf_repo.cache_waveform_repo(repo_fname);
}

/**
 * @brief sets up waveform table with given parameters, which is then updated
 * with waveform data and saved in binary
 * @param Nt_x => number of traps (smaller of the two dimensions)
 * @param Nt_y => number of traps (larger of the two dimensions)
 * @param sample_rate => waveform sampling rate
 * @param wf_duration => waveform duration
 * @param coef_x_fname => file name for file containing diffraction of AOD in
 * the X direction
 * @param coef_y_fname => file name for file containing diffraction of AOD in
 * the Y direction
 * @param is_transposed => boolean for if camera is transposed, requiring the
 * graph to also be transposed (1 is transposed, 0 not transposed)
 * @return status code of entire process
 */

Synthesis::WaveformTable
Setup::create_wf_table(size_t Nt_x, size_t Nt_y, double sample_rate,
                       double wf_duration, int waveform_length,
                       int waveform_mask, int vpp, std::string coef_x_fname,
                       std::string coef_y_fname, bool is_transposed) {

    std::string file_name = "repo." + std::to_string(Nt_x) + "_" +
                            std::to_string(Nt_y) + "_" +
                            std::to_string((int)std::round(sample_rate)) + "_" +
                            std::to_string((int)std::round(wf_duration * 1e6)) +
                            "_" + std::to_string(waveform_length) + "_" +
                            std::to_string(waveform_mask) + ".bin";
    std::string repo_path = WF_REPO_PATH(file_name);

    if (!FILE_EXISTS(repo_path)) {
        int status = create_wf_repo(Nt_x, Nt_y, sample_rate, wf_duration,
                                    waveform_length, waveform_mask, vpp,
                                    coef_x_fname, coef_y_fname);
        if (status != LLRS_OK) {
            throw std::runtime_error("Failed to create/cache waveform repo");
        }
    }
    Synthesis::WaveformRepo wf_repo(Nt_x, Nt_y, sample_rate, wf_duration,
                                    waveform_length, waveform_mask, vpp);

    wf_repo.load_waveform_repo(repo_path);

    Synthesis::WaveformTable wf_table =
        Synthesis::WaveformTable(&wf_repo, is_transposed, waveform_mask);

    return wf_table;
}
