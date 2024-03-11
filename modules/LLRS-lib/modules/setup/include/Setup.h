**
 * @brief Header file for parameter handling and waveform synthesis setup
 * @date Aug 2023
*/


#ifndef _PARAMS_H_
#define _PARAMS_H_
#include <tuple>
#include <fstream>
#include "WaveformRepo.h"
#include "WaveformTable.h"
#include "LLRS-lib/Settings.h"
#define PARAM_NUM 3

namespace Setup
{
    int read_fparams(std::string file_name, std::vector<Synthesis::WP>& params, const int limit);
    int create_wf_repo(size_t Nt_x, size_t Nt_y, double sample_rate, double wf_duration, std::string coef_x_fname, std::string coef_y_fname);
    Synthesis::WaveformTable create_wf_table(size_t Nt_x, size_t Nt_y, double sample_rate, double wf_duration, std::string coef_x_fname, std::string coef_y_fname, bool is_transposed);
}

#endif
