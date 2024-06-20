#ifndef HDF5_WRAPPER_H_
#define HDF5_WRAPPER_H_

#include "globals-not-found-exceptions.h"
#include <boost/variant.hpp>
#include <cassert>
#include <cstring>
#include <hdf5/serial/H5Cpp.h>
#include <hdf5/serial/H5Attribute.h>
#include <iostream>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <vector>

using LabscriptDictValueTypes = boost::variant<std::string, int, bool, double>;
using LabscriptDictType =
    std::unordered_map<std::string, LabscriptDictValueTypes>;

class ShotFile {

    static const std::string GLOBALS_GROUP_NAME;

    H5::H5File file;
    H5::Group globals_group;

    void assert_global_exists(std::string global_name);
    hsize_t get_global_list_value_length(std::string global_name);

  public:
    ShotFile(std::string address);

    std::vector<std::string> get_global_names();

    template <typename T>
    void get_global_value(std::string global_name, T *output) {
        assert_global_exists(global_name);

        H5::Attribute attribute =
            globals_group.openAttribute(global_name.c_str());
        attribute.read(attribute.getDataType(), output);
    }

    void get_global_dict(std::string global_name, LabscriptDictType *output);

    template <typename T>
    void get_global_list_value(std::string global_name,
                               std::vector<T> *output) {
        H5::Attribute attribute =
            globals_group.openAttribute(global_name.c_str());

        hsize_t list_size = get_global_list_value_length(global_name);

        T *result = new T[list_size];
        attribute.read(attribute.getDataType(), result);

        attribute.close();

        std::vector<T> result_vec;
        for (unsigned i = 0; i < list_size; i++) {
            result_vec.push_back(result[i]);
        }

        delete[] result;

        *output = result_vec;
    }

    void get_global_list_dict(std::string global_name, std::vector<LabscriptDictType> *output);

    void close_file();

    static LabscriptDictType convert_chars_to_labscript_dict(char *chars_dict);
};

#endif