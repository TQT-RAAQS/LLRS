#include "qdac-client.h"

QdacClient::QdacClient(std::string config) {
    auto address = QDAC_CLIENT_CONFIG(config);
    INFO << "Initializing QDAC Client with config file: " << address << "\n";
    this->configs = YAML::LoadFile(address);
    INFO << "QDAC Client configuration loaded successfully.\n";

    this->setup_client();
}

bool QdacClient::handshake() {
    try {
        auto response = this->send_string("hello");
        if (response == "hello") {
            INFO << "Handshake successful with QDAC Server.\n";
            return true;
        } else {
            ERROR << "Unexpected handshake response from QDAC Server: " << response << "\n";
            return false;
        }
    } catch (const std::exception& e) {
        ERROR << "Handshake failed: " << e.what() << "\n";
        return false;
    }
}

bool QdacClient::send_b_field(int pid_index, double b_field) {
    try {
        std::ostringstream oss;
        oss << "ramsey|" 
            << pid_index 
            << "|" 
            << std::setprecision(std::numeric_limits<double>::max_digits10) 
            << b_field;

        auto command = oss.str();
        
        auto response = this->send_string(command);
        if (response == "202") {
            INFO << "Successfully sent magnetic field value to QDAC Server: " << b_field << " T.\n";
            return true;
        } else {
            ERROR << "Failed to set magnetic field on QDAC Server. Response: " << response << "\n";
            return false;
        }
    } catch (const std::exception& e) {
        ERROR << "Error sending magnetic field to QDAC Server: " << e.what() << "\n";
        return false;
    }
}

void QdacClient::setup_client() {
    try {
        INFO << "Setting up QDAC Client...\n";

        this->context = zmq::context_t{1};

        this->socket = zmq::socket_t{this->context, zmq::socket_type::req};

        auto host = this->configs["host"].as<std::string>();
        auto port = this->configs["port"].as<int>();
        auto endpoint = std::string{"tcp://" + host + ":" + std::to_string(port)};

        this->socket.connect(endpoint);

        INFO << "Socket connected to QDAC Server at " << endpoint << "\n";
    } catch (const zmq::error_t& e) {
        ERROR << "Failed to set up QDAC Client: " << e.what() << "\n";
        throw;
    }
}

std::string QdacClient::send_string(std::string command, double timeout) {
    try {
        INFO << "Sending the command: " << command << " to QDAC Server...\n";
        this->socket.send(zmq::buffer(command), zmq::send_flags::none);

        if (timeout < 0) {
            timeout = this->configs["client_timeout"].as<double>();
        }
        this->socket.set(zmq::sockopt::rcvtimeo, static_cast<int>(timeout * 1000));

        auto reply = zmq::message_t{};
        auto res = this->socket.recv(reply, zmq::recv_flags::none);

        if (!res) {
            throw std::runtime_error("Timeout while waiting for response from QDAC Server");
        }

        return reply.to_string();
    } catch (const zmq::error_t& e) {
        ERROR << "Failed to send command " << command.substr(0, 20) << "to QDAC Server: " << e.what() << "\n";
        throw;
    }
}