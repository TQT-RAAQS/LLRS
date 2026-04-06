#ifndef _QDAC_CLIENT_H_
#define _QDAC_CLIENT_H_

#include <string>
#include <yaml-cpp/yaml.h>
#include <zmq.hpp>
#include "llrs-lib/PreProc.h"

class QdacClient {

    YAML::Node configs;
    double timeout;

    zmq::context_t context;
    zmq::socket_t socket;
    void setup_client();

    std::string send_string(std::string command, double timeout = -1);

public:
    QdacClient(std::string config);

    bool handshake();
};

#endif