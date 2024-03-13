/**
 * @brief   Wrapper for all hdf5 processing done in cpp.
 * @author  Fred S
 * @date    Feb 2024
 */

#include "h5-wrapper.hpp"
using json = nlohmann::json;
using namespace H5;

herr_t visit_groups(hid_t loc_id, const char* name, const H5L_info_t* info, void* operator_data) {
    std::vector<std::string>* group_names = static_cast<std::vector<std::string>*>(operator_data);

    // Check if the object is a group
    H5O_info_t obj_info;
    H5Oget_info_by_name(loc_id, name, &obj_info, H5P_DEFAULT);
    if (obj_info.type == H5O_TYPE_GROUP) {
        group_names->push_back(name);
    }

    return 0; // Continue iteration
}


std::map<std::string, std::vector<ModuleType>> H5Wrapper::parseSMConfig(std::string filepath){
    std::cout<<filepath<<std::endl;
    // Open the HDF5 file for reading
    H5::H5File file(filepath, H5F_ACC_RDONLY);

    // Create a map from strings to enums
    std::map<std::string, ModuleType> stringToModuleType = {
        {"LLRS", M_LLRS},
        {"CLO", M_CLO},
        {"RYDBERG", M_RYDBERG}
    };
    Group FSM_config_group = file.openGroup("/FSM_config");
    // Create a map to store the data
    std::map<std::string, std::vector<ModuleType>> data1;
        // Open the HDF5 file

    std::vector<std::string> group_names;

    H5Literate(FSM_config_group.getId(), H5_INDEX_NAME, H5_ITER_NATIVE, NULL, visit_groups, &group_names);

    for(auto group_name : group_names){

        std::string groups = "/FSM_config/" + group_name;
        // Open the group
        Group group = file.openGroup(groups);

        // Open the attribute
        Attribute attribute = group.openAttribute("data");

        // Get the dataspace of the attribute
        DataSpace dataspace = attribute.getSpace();

        // Get the number of elements in the dataspace
        hsize_t num_elements;
        dataspace.getSimpleExtentDims(&num_elements, NULL);

        // Create a buffer to hold the data
        std::vector<char*> data(num_elements);

        // Read the data into the buffer
        attribute.read(attribute.getDataType(), data.data());

        dataspace.close();
        attribute.close();
        group.close();

        // Convert char* elements to std::string
        std::vector<std::string> strings;
        for (size_t i = 0; i < num_elements; ++i) {
            strings.push_back(std::string(data[i]));
            // Free memory allocated by the HDF5 library
            free(data[i]);
        }

        // Clear the data vector to release memory
        data.clear();

        std::vector<ModuleType> values;

        for (const auto& val : strings) {
            auto it = stringToModuleType.find( val );
            values.push_back( it->second );
        }

        data1[group_name] = values;
    
    }
    file.close();
    return data1;

}

std::string H5Wrapper::convertHWConfig(std::string filepath) {

    size_t lastSlashPos = filepath.find_last_of('/');
    size_t lastDotPos = filepath.find_last_of('.');
    std::string ymlfilename = filepath.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1);
    std::string yamlFilepath = std::string(getenv("HOME")) + "/Experiment/experiment/modules/LLRS/config/" + ymlfilename + ".yml";
    
    std::ofstream yamlFile(yamlFilepath);
    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
    
    H5::H5File file(filepath, H5F_ACC_RDONLY);

    Group group = file.openGroup("/devices/AWG");

    // Iterate over attributes of the group
    for (int i = 0; i < group.getNumAttrs(); i++) {
        H5::Attribute attribute = group.openAttribute(i);
        H5std_string attributeName = attribute.getName();

        DataType attributeType = attribute.getDataType();

        if (attributeType.getClass() == H5T_STRING) {
            std::string attributeValue;
            attribute.read(attributeType, attributeValue);
            emitter << YAML::Key << attributeName << YAML::Value << attributeValue;
        } else if (attributeType.getClass() == H5T_FLOAT) {

            double numericValue;
            attribute.read(attributeType, &numericValue);
            emitter << YAML::Key << attributeName << YAML::Value << numericValue;
        } 
        else if(attributeType.getClass() == H5T_INTEGER){
            int intValue;
            attribute.read(attributeType, &intValue);
            emitter << YAML::Key << attributeName << YAML::Value << intValue;
        }
        else {
            std::cout << "Unsupported attribute data type" << std::endl;
        }

        // Close the attribute
        attribute.close();
    }


    // Iterate over datasets in the group
    for (int i = 0; i < group.getNumObjs(); i++) {
        H5std_string objname = group.getObjnameByIdx(i);
        if (group.getObjTypeByIdx(i) == H5G_DATASET) {

            DataSet dataset = group.openDataSet(objname);

            DataSpace dataspace = dataset.getSpace();

            int numElements = dataspace.getSimpleExtentNpoints();

            int* data = new int[numElements];

            dataset.read(data, PredType::NATIVE_INT);

            emitter << YAML::Key << objname << YAML::Value;
            emitter << YAML::Flow << YAML::BeginSeq;
            for (int j = 0; j < numElements; ++j) {
                emitter << data[j];
            }
            emitter << YAML::EndSeq;
            delete[] data;
        }
    }




    Group rootGroup = file.openGroup("devices/LLRS");


    for (int i = 0; i < rootGroup.getNumAttrs(); i++) {
        H5::Attribute attribute = rootGroup.openAttribute(i);
        H5std_string attributeName = attribute.getName();

        DataType attributeType = attribute.getDataType();

        if (attributeType.getClass() == H5T_STRING) {
            std::string attributeValue;
            attribute.read(attributeType, attributeValue);
            emitter << YAML::Key << attributeName << YAML::Value << attributeValue;
        } else if (attributeType.getClass() == H5T_FLOAT) {

            double numericValue;
            attribute.read(attributeType, &numericValue);
            emitter << YAML::Key << attributeName << YAML::Value << numericValue;
        } 
        else if(attributeType.getClass() == H5T_INTEGER){
            int intValue;
            attribute.read(attributeType, &intValue);
            emitter << YAML::Key << attributeName << YAML::Value << intValue;
        }
        else if(attributeType.getClass() == H5T_ENUM){
            bool boolValue;
            attribute.read(attributeType, &boolValue);
            emitter << YAML::Key << attributeName << YAML::Value << boolValue;
        }
        else {
            std::cout<<attributeType.getClass()<<std::endl;
            std::cout << "Unsupported attribute data type" << std::endl;
        }

        // Close the attribute
        attribute.close();
    }


    // Iterate over datasets in the group
    for (int i = 0; i < rootGroup.getNumObjs(); i++) {
        H5std_string objname = rootGroup.getObjnameByIdx(i);
        if (rootGroup.getObjTypeByIdx(i) == H5G_DATASET) {

            DataSet dataset = rootGroup.openDataSet(objname);

            DataSpace dataspace = dataset.getSpace();

            int numElements = dataspace.getSimpleExtentNpoints();

            int* data = new int[numElements];

            dataset.read(data, PredType::NATIVE_INT);

            emitter << YAML::Key << objname << YAML::Value;
            emitter << YAML::Flow << YAML::BeginSeq;
            for (int j = 0; j < numElements; ++j) {
                emitter << data[j];
            }
            emitter << YAML::EndSeq;
            delete[] data;
        }
    }
    rootGroup.close();

    std::vector<std::string> group_names;
    H5::Group pGroup = file.openGroup("/devices/LLRS/problem_definition");
    H5Literate(pGroup.getId(), H5_INDEX_NAME, H5_ITER_NATIVE, NULL, visit_groups, &group_names);

    emitter << YAML::Key << "problem_definition" << YAML::Value << YAML::BeginMap;
    for(auto group_name : group_names){
        Group group2 = file.openGroup("/devices/LLRS/problem_definition/"+group_name);
        emitter << YAML::Key << group_name << YAML::Value << YAML::BeginMap;
        // Iterate over attributes of the group
        for (int i = 0; i < group2.getNumAttrs(); i++) {
            H5::Attribute attribute = group2.openAttribute(i);
            H5std_string attributeName = attribute.getName();

            DataType attributeType = attribute.getDataType();
            if (attributeType.getClass() == H5T_STRING) {
                std::string attributeValue;
                attribute.read(attributeType, attributeValue);
                emitter << YAML::Key << attributeName << YAML::Value << attributeValue;
            } else if (attributeType.getClass() == H5T_FLOAT) {

                H5::DataSpace dataSpace = attribute.getSpace();
                H5::DataType dataType = attribute.getDataType();

                if(dataSpace.getSimpleExtentNdims()==1){
                    hsize_t size = dataSpace.getSimpleExtentNpoints();

                    double *data = new double[size];

                    // Read the attribute data
                    attribute.read(dataType, data);

                    emitter << YAML::Key << attributeName << YAML::Value;
                    emitter << YAML::Flow << YAML::BeginSeq;
                    for (hsize_t i = 0; i < size; ++i) {
                        emitter << data[i];
                    }
                    emitter << YAML::EndSeq;
                    delete data;
                }
                else{
                    double attributeValue;
                    attribute.read(attributeType, &attributeValue);
                    emitter << YAML::Key << attributeName << YAML::Value << attributeValue;
                }
            } 
            else if (attributeType.getClass() == H5T_INTEGER) {
                H5::DataSpace dataSpace = attribute.getSpace();
                H5::DataType dataType = attribute.getDataType();
                if(dataSpace.getSimpleExtentNdims()==1){
                    hsize_t size = dataSpace.getSimpleExtentNpoints();

                    int64_t *data = new int64_t[size];

                    // Read the attribute data
                    attribute.read(dataType, data);

                    emitter << YAML::Key << attributeName << YAML::Value;
                    emitter << YAML::Flow << YAML::BeginSeq;
                    for (hsize_t i = 0; i < size; ++i) {
                        emitter << data[i];
                    }
                    emitter << YAML::EndSeq;
                    delete data;
                }
                else{
                    int64_t attributeValue;
                    attribute.read(attributeType, &attributeValue);
                    emitter << YAML::Key << attributeName << YAML::Value << attributeValue;
                }
            }
                
            
            else if(attributeType.getClass() == H5T_ENUM){
                bool boolValue;
                attribute.read(attributeType, &boolValue);
                emitter << YAML::Key << attributeName << YAML::Value << boolValue;
            }
            else {
                std::cout << "Unsupported attribute data typeasd" << std::endl;
            }

            // Close the attribute
            attribute.close();
        }

        group2.close();
        emitter << YAML::EndMap;
    }
    emitter << YAML::EndMap;
    yamlFile << emitter.c_str();
    yamlFile.close();
    pGroup.close();
    rootGroup.close();
    group.close();
    file.close();
    
    return yamlFilepath;
}
