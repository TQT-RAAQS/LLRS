#include "JsonWrapper.h"

void Util::write_json_file(const Json::Value &value,
                           const std::string &filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        Json::StreamWriterBuilder builder;
        builder["indentation"] = "\t";
        std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
        writer->write(value, &file);
        file.close();
    } else {
        ERROR << "Error opening file: " << filename << std::endl;
    }
}

void Util::write_json_file(const nlohmann::json &value,
                           const std::string &filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << std::setw(4) << value << std::endl;
    } else {
        ERROR << "Error opening file: " << filename << std::endl;
    }
}

void convertYamlToJson(const YAML::Node &yamlNode, Json::Value &jsonValue) {
    if (yamlNode.IsScalar()) {
        if (yamlNode.IsScalar() && yamlNode.Tag() == "!int") {
            jsonValue = yamlNode.as<int>();
        } else {
            jsonValue = yamlNode.as<std::string>();
        }
    } else if (yamlNode.IsSequence()) {
        for (const auto &element : yamlNode) {
            Json::Value jsonArrayElement;
            convertYamlToJson(element, jsonArrayElement);
            jsonValue.append(jsonArrayElement);
        }
    } else if (yamlNode.IsMap()) {
        for (const auto &entry : yamlNode) {
            const std::string key = entry.first.as<std::string>();
            Json::Value jsonObjectElement;
            convertYamlToJson(entry.second, jsonObjectElement);
            jsonValue[key] = jsonObjectElement;
        }
    }
}

Json::Value Util::read_json_file(const std::string &filename) {
    std::ifstream input_file(filename);
    std::string file_contents((std::istreambuf_iterator<char>(input_file)),
                              std::istreambuf_iterator<char>());
    // Attempt to parse the content as JSON
    Json::Value root;
    Json::Reader json_reader;
    try {
        if (!json_reader.parse(file_contents, root)) {
            throw std::invalid_argument("Failed to parse JSON file");
        }
    } catch (...) {

        try {
            YAML::Node yamlNode = YAML::LoadFile(filename);

            convertYamlToJson(yamlNode, root);

        } catch (const YAML::Exception &e) {
            ERROR << "Failed to parse JSON or YAML file: " << e.what()
                  << std::endl;
            return Json::Value();
        }
    }

    return root;
}

std::vector<int32_t> Util::vector_transform(Json::Value val) {
    std::vector<int32_t> ret;
    std::transform(val.begin(), val.end(), std::back_inserter(ret),
                   [](const Json::Value &e) -> int32_t {
                       return std::stoi(e.asString());
                   });
    return ret;
}
Util::JsonWrapper::JsonWrapper(const std::string &filename)
    : json_data(read_json_file(filename)) {}

Util::JsonWrapper::JsonWrapper(const std::string &filename, std::string name)
    : json_data(read_json_file(filename)), name(name) {}

Util::JsonWrapper::JsonWrapper(Json::Value json_data, std::string name)
    : json_data(json_data), name(name) {}

Json::Value Util::JsonWrapper::get_json_data() const { return json_data; }

int Util::JsonWrapper::read_problem_Nt_x() const {
    if (json_data.isMember("problem_definition") &&
        json_data["problem_definition"].isMember("problem_params") &&
        json_data["problem_definition"]["problem_params"].isMember("Nt_x")) {
        return stoi(json_data["problem_definition"]["problem_params"]["Nt_x"]
                        .asString());
    } else {
        throw std::invalid_argument(
            "Missing field: problem_definition.problem_params.Nt_x");
    }
}

int Util::JsonWrapper::read_problem_Nt_y() const {
    if (json_data.isMember("problem_definition") &&
        json_data["problem_definition"].isMember("problem_params") &&
        json_data["problem_definition"]["problem_params"].isMember("Nt_y")) {
        return stoi(json_data["problem_definition"]["problem_params"]["Nt_y"]
                        .asString());
    } else {
        throw std::invalid_argument(
            "Missing field: problem_definition.problem_params.Nt_y");
    }
}

int Util::JsonWrapper::read_problem_num_target() const {
    if (json_data.isMember("problem_definition") &&
        json_data["problem_definition"].isMember("problem_params") &&
        json_data["problem_definition"]["problem_params"].isMember(
            "num_target")) {
        return stoi(
            json_data["problem_definition"]["problem_params"]["num_target"]
                .asString());
    } else {
        throw std::invalid_argument(
            "Missing field: problem_definition.problem_params.num_target");
    }
}

double Util::JsonWrapper::read_problem_alpha() const {
    if (json_data.isMember("problem_definition") &&
        json_data["problem_definition"].isMember("loss_atom_params") &&
        json_data["problem_definition"]["loss_atom_params"].isMember(
            "p_alpha")) {
        return stod(
            json_data["problem_definition"]["loss_atom_params"]["p_alpha"]
                .asString());
    } else {
        throw std::invalid_argument(
            "Missing field: problem_definition.loss_atom_params.alpha");
    }
}
double Util::JsonWrapper::read_problem_nu() const {
    if (json_data.isMember("problem_definition") &&
        json_data["problem_definition"].isMember("loss_atom_params") &&
        json_data["problem_definition"]["loss_atom_params"].isMember("p_nu")) {
        return stod(json_data["problem_definition"]["loss_atom_params"]["p_nu"]
                        .asString());
    } else {
        throw std::invalid_argument(
            "Missing field: problem_definition.loss_atom_params.nu");
    }
}

double Util::JsonWrapper::read_problem_lifetime() const {
    if (json_data.isMember("problem_definition") &&
        json_data["problem_definition"].isMember("loss_atom_params") &&
        json_data["problem_definition"]["loss_atom_params"].isMember(
            "t_lifetime")) {
        return stod(
            json_data["problem_definition"]["loss_atom_params"]["t_lifetime"]
                .asString());
    } else {
        throw std::invalid_argument(
            "Missing field: problem_definition.loss_atom_params.lifetime");
    }
}

int Util::JsonWrapper::read_problem_num_trials() const {
    if (json_data.isMember("problem_definition") &&
        json_data["problem_definition"].isMember("problem_params") &&
        json_data["problem_definition"]["problem_params"].isMember(
            "num_trials_per_problem")) {
        return stoi(json_data["problem_definition"]["problem_params"]
                             ["num_trials_per_problem"]
                                 .asString());
    } else {
        throw std::invalid_argument(
            "Missing field: "
            "problem_definition.problem_params.num_trials_per_problem");
    }
}

int Util::JsonWrapper::read_problem_num_reps() const {
    if (json_data.isMember("problem_definition") &&
        json_data["problem_definition"].isMember("problem_params") &&
        json_data["problem_definition"]["problem_params"].isMember(
            "repetitions_per_trial")) {
        return stoi(json_data["problem_definition"]["problem_params"]
                             ["repetitions_per_trial"]
                                 .asString());
    } else {
        throw std::invalid_argument(
            "Missing field: "
            "problem_definition.problem_params.repetitions_per_trial");
    }
}

std::string Util::JsonWrapper::read_problem_target_config_label() const {
    if (json_data.isMember("problem_definition") &&
        json_data["problem_definition"].isMember("problem_params") &&
        json_data["problem_definition"]["problem_params"].isMember(
            "target_config_label")) {
        return json_data["problem_definition"]["problem_params"]
                        ["target_config_label"]
                            .asString();
    } else {
        throw std::invalid_argument(
            "Missing field: "
            "problem_definition.problem_params.target_config_label");
    }
}

std::vector<int> Util::JsonWrapper::read_problem_target_config() const {
    std::vector<int> target_config;
    if (json_data.isMember("problem_definition") &&
        json_data["problem_definition"].isMember("problem_params") &&
        json_data["problem_definition"]["problem_params"].isMember(
            "target_config")) {

        const Json::Value &target_config_json =
            json_data["problem_definition"]["problem_params"]["target_config"];
        if (target_config_json.isArray()) {
            for (Json::ArrayIndex i = 0; i < target_config_json.size(); ++i) {
                target_config.push_back(stoi(target_config_json[i].asString()));
            }
        }
    }
    if (target_config.empty()) {
        throw std::invalid_argument(
            "Missing or invalid field: "
            "problem_definition.problem_params.target_config");
    }

    return target_config;
}

std::string Util::JsonWrapper::read_problem_id() const {
    if (json_data.isMember("problem_definition") &&
        json_data["problem_definition"].isMember("uuid")) {
        return json_data["problem_definition"]["uuid"].asString();
    } else {
        throw std::invalid_argument("Missing field: problem_definition.uuid");
    }
}

std::string Util::JsonWrapper::read_problem_algo() const {
    if (json_data.isMember("problem_definition") &&
        json_data["problem_definition"].isMember("problem_params") &&
        json_data["problem_definition"]["problem_params"].isMember(
            "algorithm")) {
        return json_data["problem_definition"]["problem_params"]["algorithm"]
            .asString();
    } else {
        throw std::invalid_argument(
            "Missing field: problem_definition.problem_params.algorithm");
    }
}

std::string Util::JsonWrapper::read_experiment_psf_path() const {
    if (json_data.isMember("problem_definition") &&
        json_data["problem_definition"].isMember("experiment_params") &&
        json_data["problem_definition"]["experiment_params"].isMember("psf")) {
        return json_data["problem_definition"]["experiment_params"]["psf"]
            .asString();
    } else {
        throw std::invalid_argument(
            "Missing field: problem_definition.experiment_params.psf");
    }
}

std::string Util::JsonWrapper::read_experiment_coefx_path() const {
    if (json_data.isMember("problem_definition") &&
        json_data["problem_definition"].isMember("experiment_params") &&
        json_data["problem_definition"]["experiment_params"].isMember(
            "coef_x")) {
        return json_data["problem_definition"]["experiment_params"]["coef_x"]
            .asString();
    } else {
        throw std::invalid_argument(
            "Missing field: problem_definition.experiment_params.coef_x");
    }
}

std::string Util::JsonWrapper::read_experiment_coefy_path() const {
    if (json_data.isMember("problem_definition") &&
        json_data["problem_definition"].isMember("experiment_params") &&
        json_data["problem_definition"]["experiment_params"].isMember(
            "coef_y")) {
        return json_data["problem_definition"]["experiment_params"]["coef_y"]
            .asString();
    } else {
        throw std::invalid_argument(
            "Missing field: problem_definition.experiment_params.coef_y");
    }
}

double Util::JsonWrapper::read_experiment_threshold() const {
    if (json_data.isMember("problem_definition") &&
        json_data["problem_definition"].isMember("experiment_params") &&
        json_data["problem_definition"]["experiment_params"].isMember(
            "detection_threshold")) {
        return stod(json_data["problem_definition"]["experiment_params"]
                             ["detection_threshold"]
                                 .asString());
    } else {
        throw std::invalid_argument(
            "Missing field: "
            "problem_definition.experiment_params.detection_threshold");
    }
}

int Util::JsonWrapper::read_experiment_roi_width() const {
    if (json_data.isMember("problem_definition") &&
        json_data["problem_definition"].isMember("experiment_params") &&
        json_data["problem_definition"]["experiment_params"].isMember(
            "roi_width")) {
        return stoi(
            json_data["problem_definition"]["experiment_params"]["roi_width"]
                .asString());

    } else {
        throw std::invalid_argument(
            "Missing field: problem_definition.experiment_params.roi_width");
    }
}

int Util::JsonWrapper::read_experiment_roi_height() const {
    if (json_data.isMember("problem_definition") &&
        json_data["problem_definition"].isMember("experiment_params") &&
        json_data["problem_definition"]["experiment_params"].isMember(
            "roi_height")) {
        return stoi(
            json_data["problem_definition"]["experiment_params"]["roi_height"]
                .asString());
    } else {
        throw std::invalid_argument(
            "Missing field: problem_definition.experiment_params.roi_height");
    }
}
Json::Value Util::JsonWrapper::get_field(const std::string &field) const {
    if (json_data.isMember(field)) {
        return json_data[field];
    } else {
        throw std::invalid_argument("Missing field in " + name + ": " + field);
    }
}

int Util::JsonWrapper::get_int(const std::string &field) const {
    Json::Value json_val = this->get_field(field);
    if (json_val.isInt()) {
        return json_val.asInt();
    } else {
        throw std::invalid_argument("Field " + name + " in " + field +
                                    "is not an integer");
    }
}
double Util::JsonWrapper::get_double(const std::string &field) const {
    Json::Value json_val = this->get_field(field);
    if (json_val.isDouble()) {
        return json_val.asDouble();
    } else {
        throw std::invalid_argument("Field " + field + " in " + name +
                                    "is not a real number");
    }
}

std::string Util::JsonWrapper::get_string(const std::string &field) const {
    Json::Value json_val = this->get_field(field);
    if (json_val.isString()) {
        return json_val.asString();
    } else {
        throw std::invalid_argument("Field " + field + " in " + name +
                                    "is not a string");
    }
}

std::vector<std::string>
Util::JsonWrapper::get_str_arr(const std::string &field) const {
    Json::Value json_val = this->get_field(field);
    std::vector<std::string> result;
    if (json_val.isArray()) {
        for (Json::Value element : json_val) {
            if (element.isString()) {
                result.push_back(element.asString());
            } else {
                throw std::invalid_argument("Field " + field + " in " + name +
                                            "is not an array of strings");
            }
        }
    } else {
        throw std::invalid_argument("Field " + field + " in " + name +
                                    "is not an array of strings");
    }
    return result;
}
Util::IOWrapper::IOWrapper(const std::string &config_file,
                           const std::string &param_file) {
    JsonWrapper config = JsonWrapper(read_json_file(config_file), config_file);
    experiment_config =
        JsonWrapper(config.get_field("experiment_config"), "experiment_config");
    operational_config = JsonWrapper(config.get_field("operational_config"),
                                     "operational_config");
    problem_config =
        JsonWrapper(config.get_field("problem_config"), "problem_config");
    problem_param = JsonWrapper(read_json_file(param_file), "problem_param");
}

std::vector<bool> Util::IOWrapper::b64_to_bool_arr(const std::string &b64,
                                                   int len) const {
    const std::string b64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<bool> result;
    for (char c : b64) {
        if (c == '=')
            break;
        int value = b64_chars.find(c);
        if (value == std::string::npos) {
            throw std::invalid_argument(b64 + " is not valid base64 string");
        }
        for (int i = 5; i >= 0; --i) {
            result.push_back(value & (1 << i));
            if (result.size() >= len)
                break;
        }
    }
    return result;
}

std::string
Util::IOWrapper::bool_arr_to_b64(const std::vector<bool> &bool_arr) const {
    const std::string b64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string b64 = "";
    for (int i = 0; i < bool_arr.size(); i += 6) {
        int value = 0;
        for (int j = 0; j < 6; ++j) {
            int bit_val = (i + j >= bool_arr.size()) ? 0 : bool_arr[i + j];
            value += bit_val << (5 - j);
        }
        b64 += b64[value];
    }
    while (b64.length() % 4 != 0) {
        b64 += '=';
    }

    return b64;
}
