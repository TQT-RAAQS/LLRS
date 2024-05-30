#include "shot-file.h"

const std::string ShotFile::GLOBALS_GROUP_NAME = "/globals";

ShotFile::ShotFile(std::string address)
    : file(address.c_str(), H5F_ACC_RDONLY) {
    globals_group = file.openGroup(ShotFile::GLOBALS_GROUP_NAME);
}

void ShotFile::close_file() {
    file.close();
}

void ShotFile::assert_global_exists(std::string global_name) {
    if (!globals_group.attrExists(global_name.c_str())) {
        std::string error_1 = "The globals '";
        std::string error_2 = "' does not exist in the shot file '";
        std::string error_3 = "'.";

        throw GlobalsNotFoundException(error_1 + global_name + error_2 +
                                       file.getFileName() + error_3);
    }
}

hsize_t ShotFile::get_global_list_value_length(std::string global_name) {
    H5::Attribute attribute = globals_group.openAttribute(global_name.c_str());
    hsize_t list_size;
    attribute.getSpace().getSimpleExtentDims(&list_size);

    return list_size;
}

void ShotFile::get_global_dict(std::string global_name,
                               LabscriptDictType *output) {
    char *dict_str;
    get_global_value(global_name, &dict_str);
    *output = ShotFile::convert_chars_to_labscript_dict(dict_str);
}

std::vector<std::string> ShotFile::get_global_names() {
    std::vector<std::string> attribute_names;

    auto iter_get_attribute_name =
        [](H5::H5Location &loc, H5std_string attr_name, void *operator_data) {
            std::vector<std::string> *attributes_vector =
                static_cast<std::vector<std::string> *>(operator_data);
            attributes_vector->push_back(attr_name);
        };

    globals_group.iterateAttrs(iter_get_attribute_name, NULL, &attribute_names);

    return attribute_names;
}

LabscriptDictType ShotFile::convert_chars_to_labscript_dict(char *chars_dict) {
    std::string str_dict(chars_dict);

    assert(str_dict.length() > 2);
    assert(str_dict.at(0) == '{');
    assert(str_dict.at(str_dict.length() - 1) == '}');

    LabscriptDictType output;
    str_dict = str_dict.substr(1, str_dict.length() - 2);

    std::vector<std::string> keyValuePairList;
    while (true) {
        unsigned long commaPosition = str_dict.find(
            ','); // It is important for this to be an unsigned long for the
                  // comparison with std::string::npos to work
        if (commaPosition == std::string::npos) {
            keyValuePairList.push_back(str_dict);
            break;
        }

        keyValuePairList.push_back(str_dict.substr(0, commaPosition));
        str_dict = str_dict.substr(commaPosition + 2);
    }

    int keyValuePairListSize = keyValuePairList.size();
    unsigned long colonPosition;
    std::string key, str_value;
    LabscriptDictValueTypes value;
    for (int i = 0; i < keyValuePairListSize; i++) {
        std::string keyValuePair = keyValuePairList.at(i);

        colonPosition = keyValuePair.find(':');
        key = keyValuePair.substr(1, colonPosition - 2);
        str_value = keyValuePair.substr(colonPosition + 2);

        if (str_value.find('\'') != std::string::npos) {
            value = str_value.substr(1, str_value.length() - 2);
        } else if (str_value.find('.') != std::string::npos || 
                   str_value.find('e') != std::string::npos || 
                   str_value.find('E') != std::string::npos) {
            // Conversion of a string to double, taking into account that the string could be in scientific format.
            std::istringstream iss(str_value);
            double temp_double;
            iss >> temp_double;

            value = temp_double;
        } else {
            value = std::stol(str_value);
        }

        output.emplace(key, value);
    }

    return output;
}