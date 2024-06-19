#ifndef LLCS_HANDLER_HPP_
#define LLCS_HANDLER_HPP_

#include "server.hpp"
#include <chrono>
#include <future>
#include <mutex>

class Handler {
    Server server;
    std::future<void> listen_thread;
    void async_listen();

    std::mutex processingMutex;
    std::mutex requestMutex;
    enum State {
        WAITING,
        RECEIVED_HDF5_FILE_PATH,
        RECEIVED_DONE,
    } request = WAITING;
    std::string hdf5_file_path;
    bool processing = false;

  public:
    Handler();
    Handler(Handler &) = delete;
    Handler &operator=(Handler &) = delete;
    ~Handler();
    void start_listening();
    std::string get_hdf5_file_path();
    void send_done();
    void wait_for_done();
};

#endif
