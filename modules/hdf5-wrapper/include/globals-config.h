#ifndef HDF5_GENERIC_CONFIG_H_
#define HDF5_GENERIC_CONFIG_H_

#include "shot-file.h"
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

enum LabscriptType {
    VALUE,
    DICT,
    LIST_OF_INT,
    LIST_OF_DOUBLE,
    LIST_OF_BOOL,
    LIST_OF_CHARS,
    LIST_OF_DICT
};

using ConfigDescription = std::tuple<std::string, void *, LabscriptType>;

class GlobalsConfig {

    std::vector<ConfigDescription> globals_list;
    void read_from_shot_file();

  protected:
    ShotFile shotfile;
    std::string shot_address;

    GlobalsConfig(
        const std::string& shot_address,
        std::vector<ConfigDescription> globals_list
    )
        : globals_list(std::move(globals_list)),
          shotfile(shot_address),
          shot_address(shot_address) {

        read_from_shot_file();
    }
};

#endif