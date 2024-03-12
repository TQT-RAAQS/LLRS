/**
 * @brief
 * @date Nov 2023
*/

#ifndef PROBLEM_H
#define PROBLEM_H

#include <string>
#include <vector>
#include <cctype>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <jsoncpp/json/json.h>
#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>
#include "LLRS-lib/Settings.h"
#include "LLRS-lib/PreProc.h"


namespace Util {

    void write_json_file(const Json::Value& value, const std::string& filename);
    Json::Value read_json_file(const std::string& filename);
    std::vector<int32_t> vector_transform(Json::Value val);

    class JsonWrapper {
        Json::Value json_data;
        std::string name;
    public:
        JsonWrapper(const std::string& filename, std::string name);
        JsonWrapper(const std::string& filename);
        JsonWrapper(Json::Value json_data, std::string name);
        JsonWrapper() = default;
        Json::Value get_json_data() const;

        int read_problem_Nt_x() const;
        int read_problem_Nt_y() const;
        int read_problem_num_target() const;
        int read_problem_num_trials() const;
        int read_problem_num_reps() const;
		double read_problem_alpha() const;
		double read_problem_nu() const;
		double read_problem_lifetime() const;
        std::string read_problem_target_config_label() const;
        std::vector<int> read_problem_target_config() const;
        std::string read_problem_id() const;
        std::string read_problem_algo() const;
        std::string read_experiment_psf_path() const;
        std::string read_experiment_coefx_path() const;
        std::string read_experiment_coefy_path() const;
        double read_experiment_threshold() const;
        int read_experiment_roi_width() const;
        int read_experiment_roi_height() const;
        void write_json_file(const Json::Value& value, const std::string& filename) const;
        // Added methods from the second class
        Json::Value get_field(const std::string& name) const;
        int get_int(const std::string& name) const;
        double get_double(const std::string& name) const;
        std::string get_string(const std::string& name) const;
        std::vector<std::string> get_str_arr(const std::string& name) const;
    };

    class IOWrapper {
    public:
        IOWrapper(const std::string& config_file, const std::string& param_file);

        std::vector<bool> b64_to_bool_arr(const std::string& b64, int len) const;
        std::string bool_arr_to_b64(const std::vector<bool>& bool_arr) const;
        JsonWrapper experiment_config, operational_config, problem_config, problem_param;

    };

} // namespace Util

#endif // PROBLEM_H
