#ifndef SERVER_HPP_
#define SERVER_HPP_

#include "llcs/common.hpp"
#include <fstream>
#include <hdf5/serial/H5Cpp.h>
#include <nlohmann/json.hpp>
#include <zmq.hpp>

class Server {
    zmq::context_t context;
    zmq::socket_t socket;
    int listen_timeout = 60000; // ms
  public:
    Server();
    ~Server();
    bool send(const std::string &string);
    int listen(std::string &reqeustStr);
    void set_listen_timeout(int timeout);
    
    
    // TO BE MOVED TO LLCS WRAPPER:
    int llcs_handler();
    std::string metadata_file_path;
    std::string config_file_path;
    void setMetadataAddress(std::string &requestStr);
    std::string get_metadata_file_path() { return metadata_file_path; }
    std::string get_config_file_path() { return config_file_path; }
};

std::string adjust_address(std::string filename);

#endif
