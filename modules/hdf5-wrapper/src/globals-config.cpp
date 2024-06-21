#include "globals-config.h"

void GlobalsConfig::read_from_shot_file() {
    std::string global_name;

    void *var;
    int globals_list_size = globals_list.size();
    LabscriptType labscriptType;

    LabscriptDictType **dict_ptr_ptr;
    std::vector<int> **vec_ptr_ptr_int;
    std::vector<double> **vec_ptr_ptr_double;
    std::vector<bool> **vec_ptr_ptr_bool;
    std::vector<char *> **vec_ptr_ptr_chars;
    std::vector<LabscriptDictType> **vec_ptr_ptr_dict;
    std::vector<char *> dict_cstr_vec;
    for (int i = 0; i < globals_list_size; i++) {
        std::tie(global_name, var, labscriptType) = globals_list.at(i);

        switch (labscriptType) {
        case LabscriptType::VALUE:
            shotfile.get_global_value(global_name, var);
            break;
        case LabscriptType::DICT:
            dict_ptr_ptr = static_cast<LabscriptDictType **>(var);
            *dict_ptr_ptr = new LabscriptDictType();
            shotfile.get_global_dict(global_name, *dict_ptr_ptr);
            break;
        case LabscriptType::LIST_OF_INT:
            vec_ptr_ptr_int = static_cast<std::vector<int> **>(var);
            *vec_ptr_ptr_int = new std::vector<int>();
            shotfile.get_global_list_value(global_name, *vec_ptr_ptr_int);
            break;
        case LabscriptType::LIST_OF_DOUBLE:
            vec_ptr_ptr_double = static_cast<std::vector<double> **>(var);
            *vec_ptr_ptr_double = new std::vector<double>();
            shotfile.get_global_list_value(global_name, *vec_ptr_ptr_double);
            break;
        case LabscriptType::LIST_OF_BOOL:
            vec_ptr_ptr_bool = static_cast<std::vector<bool> **>(var);
            *vec_ptr_ptr_bool = new std::vector<bool>();
            shotfile.get_global_list_value(global_name, *vec_ptr_ptr_bool);
            break;
        case LabscriptType::LIST_OF_CHARS:
            vec_ptr_ptr_chars = static_cast<std::vector<char *> **>(var);
            *vec_ptr_ptr_chars = new std::vector<char *>();
            shotfile.get_global_list_value(global_name, *vec_ptr_ptr_chars);
            break;
        case LabscriptType::LIST_OF_DICT:
            vec_ptr_ptr_dict =
                static_cast<std::vector<LabscriptDictType> **>(var);
            *vec_ptr_ptr_dict = new std::vector<LabscriptDictType>();
            shotfile.get_global_list_dict(global_name, *vec_ptr_ptr_dict);

            break;
        default:
            std::cout << "shoot" << std::endl;
            break;
        }
    }
}