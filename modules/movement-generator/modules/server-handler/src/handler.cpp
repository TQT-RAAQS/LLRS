#include "handler.hpp"

Handler::Handler() {
    server.set_listen_timeout(300000); // wait for 5 minutes for each request
}

Handler::~Handler() {
    stop_listening();
}

void Handler::start_listening() {
    listen = true;
    auto reset_result = std::async(
        std::launch::async, &Handler::async_listen, this);
}

void Handler::async_listen() {
    std::string requestStr;
    while (listen) {
        int rc = server.listen(requestStr);
        if (rc == -1) {
            continue;
        }
        if (requestStr == "hello") {
            server.send("hello");
            continue;
        } else if (requestStr == "abort") {
            received_abort = true;
            listen = false;
            continue;
        } else if (requestStr == "done") {
            server.send("ok");
            received_done = true;
            server.listen(requestStr);
            continue;
        } else if (requestStr.substr(requestStr.length() - 3, 3) == ".h5") {
            hdf5_file_path = adjust_address(requestStr);
            obtained_hdf5_file_path = true;
            server.send("ok");
            server.listen(requestStr); 
            continue;
        }
    }
}

std::string Handler::get_hdf5_file_path() {
    while (!obtained_hdf5_file_path) {
        sleep(2000);
    }
    obtained_hdf5_file_path = false; 
    return hdf5_file_path;
}

void Handler::wait_for_done() {
    while (!received_done) {
        sleep(2000);
    }
    received_done = false;
    return;
};
