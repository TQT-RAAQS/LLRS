#ifndef QDAC_CLIENT_H_
#define QDAC_CLIENT_H_

#include <string>
#include <yaml-cpp/yaml.h>
#include <zmq.hpp>
#include <sstream>
#include <iomanip>
#include <limits>
#include "llrs-lib/PreProc.h"

class QdacClient {

    YAML::Node configs;
    double timeout;

    zmq::context_t context;
    zmq::socket_t socket;
    void setup_client();

    std::string send_string(std::string command, double timeout = -1.0);

public:
    QdacClient(std::string config);

    bool handshake();
    bool send_b_field(int pid_index, double b_field);
};

#endif