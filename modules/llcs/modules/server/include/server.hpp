#ifndef SERVER_HPP_
#define SERVER_HPP_

#include "llcs/common.hpp"
#include <fstream>
#include <hdf5/serial/H5Cpp.h>
#include <nlohmann/json.hpp>
#include <zmq.hpp>

class Server {

  private:
    zmq::context_t context;
    zmq::socket_t socket;
    std::string metadata_file_path;
    std::string config_file_path;

  protected:
  public:
    Server();
    ~Server();
    bool send(const std::string &string);
    int listen();
    int IdleTransition();
    void send_metadata(std::string filename);
    void getMetadataAddress(std::string &requestStr);
    std::string get_metadata_file_path();
    std::string get_config_file_path();
};

#endif
