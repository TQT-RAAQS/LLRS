/**
 * @brief Header file for parameter handling and waveform synthesis setup
 * @date Aug 2023
 */

#ifndef PARAMS_H_
#define PARAMS_H_
#include "WaveformRepo.h"
#include "WaveformTable.h"
#include "llrs-lib/PreProc.h"
#include "llrs-lib/Settings.h"
#include <fstream>
#include <tuple>
#define PARAM_NUM 3

namespace Setup {
int read_fparams(std::string file_name, std::vector<Synthesis::WP> &params,
                 const int limit);
int create_wf_repo(size_t Nt_x, size_t Nt_y, double sample_rate,
                   double wf_duration, int wavefrom_length, int waveform_mask,
                   int vpp, , std::string coef_x_fname,
                   std::string coef_y_fname);
Synthesis::WaveformTable create_wf_table(size_t Nt_x, size_t Nt_y,
                                         double sample_rate, double wf_duration,
                                         int wavefrom_length, int waveform_mask,
                                         int vpp, std::string coef_x_fname,
                                         std::string coef_y_fname,
                                         bool is_transposed);
} // namespace Setup

#endif
