#include "utility.h"

/**
 * @brief Sums the elements of a vector to find the number of the atoms
 *
 * @param atom_config The vector containing the configuration of the atoms
 * @return int Number of atoms in the configuration
 */
int Util::count_num_atoms(std::vector<int32_t> atom_config) {
    return std::accumulate(atom_config.begin(), atom_config.end(),
                           decltype(atom_config)::value_type(0));
}

/**
 * @brief Check if all target atoms exists in current trap
 * Loop through the target and current configuration and check if all the target
 * atoms exist in the current trap
 * @param current
 * @param target
 * @return true
 * @return false
 */
bool Util::target_met(std::vector<int32_t> current,
                      std::vector<int32_t> target) {
    for (size_t trap = 0; trap < target.size() && trap < current.size();
         ++trap) {
        if (target[trap] && !current[trap]) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Get the algo enum object from the string name
 *
 * @param name
 * @return Algo
 * @throw std::invalid_argument if the name is not a valid algorithm name
 */

Reconfig::Algo Util::get_algo_enum(std::string name) {
    if (name == "liblin_exact_1d.so") {
        return Reconfig::LINEAR_EXACT_1D;
    } else if (name == "liblin_exact_1d_cpu_v2.so") {
        return Reconfig::LINEAR_EXACT_V2_1D;
    } else if (name == "libredrec_v2.so") {
        return Reconfig::REDREC_CPU_V2_2D;
    } else if (name == "libredrec_cpu_v3.so") {
        return Reconfig::REDREC_CPU_V3_2D;
    } else if (name == "libaro_cpu.so") {
        return Reconfig::ARO_CPU_2D;
    } else if (name == "libredrec_gpu_v3.so") {
        return Reconfig::REDREC_GPU_V3_2D;
    } else if (name == "libbird_cpu.so") {
        return Reconfig::BIRD_CPU_2D;
    } else {
        throw std::invalid_argument(
            "Algorithm not supported"); /// indicates that some other algorithm
                                        /// has been provided which is not
                                        /// supported
    }
}

/**
 * @brief Prints the configuration vector
 * @param x coordinate definition
 * @param y coordinate definition
 * @param current_config binary vector indicating the presence of atoms
 */
void printVec(int x, int y, std::vector<int32_t> &current_config) {
    for (int j = 0; j < y; j++) {
        for (int i = 0; i < x; i++) {
            std::cout << current_config[i + x * j] << ", ";
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Parse a string representing a pretty-printed vector ("1, 2, 3, 4")
 into a vector of integers
 * @param str The string to parse
 * @return std::vector<int> The parsed vector

*/
std::vector<int> Util::parse_pretty_vector(const std::string &str) {
    std::vector<int> result;
    std::stringstream ss(str);
    std::string element_str;

    while (std::getline(ss, element_str, ',')) {
        int element;
        element_str.erase(
            std::remove_if(element_str.begin(), element_str.end(), ::isspace),
            element_str.end());
        std::istringstream(element_str) >> element;
        result.push_back(element);
    }

    return result;
}
