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
        WAITING,                         // 0
        RECEIVED_HDF5_FILE_PATH,         // 1
        RECEIVED_RESET_REQUEST,          // 2
        RECEIVED_PSF_RESET_REQUEST,      // 3
        RECEIVED_WAVEFORM_RESET_REQUEST, // 4
        RECEIVED_AWG_RESET_REQUEST,      // 5
        RECEIVED_DONE                    // 6
    } request = WAITING;
    std::string hdf5_file_path;
    std::string llrs_config_file;
    bool processing = false;

  public:
    Handler();
    Handler(Handler &) = delete;
    Handler &operator=(Handler &) = delete;
    ~Handler();
    void start_listening();
    uint get_request();
    std::string get_hdf5_file_path();
    std::string get_llrs_config_file();
    void send_done();
    void send_200();
    void wait_for_done();
};

#endif