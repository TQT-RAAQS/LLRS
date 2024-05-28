#ifndef MOVE_GEN_HANDLER_HPP_
#define MOVE_GEN_HANDLER_HPP_

#include "server.hpp"
#include <future>
#include <unistd.h>


class Handler {
    Server server;
    std::string hdf5_file_path;
    bool listen = false;
    bool obtained_hdf5_file_path = false;
    bool received_done = false;
    bool received_abort = false;
    void async_listen();

  public:
    Handler();
    Handler(Handler &) = delete;
    Handler& operator=(Handler &) = delete;
    ~Handler();
    void start_listening();
    void stop_listening() {listen = false;};
    std::string get_hdf5_file_path();
    bool get_abort() {return received_abort;};
    void send_done() {server.send("done");}
    void wait_for_done();
};

#endif
