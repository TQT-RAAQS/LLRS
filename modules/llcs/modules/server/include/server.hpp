#ifndef SERVER_HPP
#define SERVER_HPP

#include "llcs/common.hpp"
#include <zmq.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <hdf5/serial/H5Cpp.h>

class Server{

    private:
        zmq::context_t context; 
        zmq::socket_t socket;
        std::string metadata_file_path;
        std::string config_file_path;


    protected:
    public:
        Server();
        ~Server();
        bool send(const std::string& string);
        int listen();
        int IdleTransition();
        void send_metadata(std::string filename);
        void getMetadataAddress(std::string &requestStr);
        std::string get_metadata_file_path();
        std::string get_config_file_path();
};

#endif
