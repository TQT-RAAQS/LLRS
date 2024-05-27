#ifndef HDF5_GENERIC_CONFIG_H_
#define HDF5_GENERIC_CONFIG_H_

#include <shot_file.h>
#include <utility>
#include <vector>
#include <string>
#include <iostream>
#include <tuple>

enum LabscriptType {
    VALUE,
    DICT,
    LIST_OF_LONG,
    LIST_OF_DOUBLE,
    LIST_OF_BOOL,
    LIST_OF_CHARS,
    LIST_OF_DICT
};

using ConfigDescription = std::tuple<std::string, void*, LabscriptType>;

class GlobalsConfig {

    std::vector<ConfigDescription> globals_list;
    void read_from_shot_file(ShotFile shotfile);

protected:

    GlobalsConfig(ShotFile shotfile, std::vector<ConfigDescription> globals_list) : globals_list(globals_list) {
        read_from_shot_file(shotfile);
    }

};

#endif