#ifndef _UTILITY_H_
#define _UTILITY_H_
/// Utility submodule that includes handy functions used in different parts of the module

#include <string>
#include <numeric>
#include "Solver.h"

namespace Util {

/**
 * @brief Get the algo enum object from the string name corresponding the filename
 * @param name 
 * @return Algo 
 * @throw std::invalid_argument if the name is not a valid algorithm name
 */
Reconfig::Algo get_algo_enum(std::string name);


/**
 * @brief Sums the elements of a vector to find the number of atoms
 * @param atom_config The vector containing the configuration of the atoms
 * @return int Number of atoms in the configuration 
 */
int count_num_atoms(std::vector<int32_t>  atom_config);      


/**
 * @brief Check if all target atoms exists in current trap
 * Loop through the target and current configuration and check if all the target atoms exist in the current trap 
 * @param current 
 * @param target 
 * @return true 
 * @return false 
 */
bool target_met(std::vector<int32_t> current, std::vector<int32_t> target);


/**
 * @brief Prints the configuration vector
 * @param x coordinate definition
 * @param y coordinate definition
 * @param current_config binary vector indicating the presence of atoms
*/
void printVec(int x, int y, std::vector<int32_t> &current_config);

/**
 * @brief Parse a string representing a pretty-printed vector ("1, 2, 3, 4") into a vector of integers
 * @param str The string to parse
 * @return std::vector<int> The parsed vector

*/
std::vector<int> parse_pretty_vector(const std::string& str);

}



#endif
