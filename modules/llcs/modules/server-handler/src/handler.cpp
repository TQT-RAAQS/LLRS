#include "handler.hpp"

Handler::Handler() {
    server.set_listen_timeout(300000); // wait for 5 minutes for each request
}

Handler::~Handler() {}

void Handler::start_listening() {
    listen_thread =
        std::async(std::launch::async, &Handler::async_listen, this);
}

void Handler::async_listen() {
    std::string requestStr;
    while (true) {
        server.listen(requestStr);
        if (requestStr == "hello") {
            server.send("hello");
            continue;
        } else if (requestStr == "done") {
            server.listen(requestStr);
            {
                std::lock_guard<std::mutex> lock(requestMutex);
                request = RECEIVED_DONE;
            }
            {
                std::lock_guard<std::mutex> lock(processingMutex);
                processing = true;
            }
            while (true) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                bool flag = false;
                {
                    std::lock_guard<std::mutex> lock(processingMutex);
                    if (!processing)
                        flag = true;
                }
                if (flag)
                    break;
            }
            continue;
        } else if (requestStr.substr(requestStr.length() - 3, 3) == ".h5") {
            server.send("ok");
            server.listen(requestStr);
            {
                std::lock_guard<std::mutex> lock(requestMutex);
                hdf5_file_path = adjust_address(requestStr);
                request = RECEIVED_HDF5_FILE_PATH;
            }
            {
                std::lock_guard<std::mutex> lock(processingMutex);
                processing = true;
            }
            while (true) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                bool flag = false;
                {
                    std::lock_guard<std::mutex> lock(processingMutex);
                    if (!processing)
                        flag = true;
                }
                if (flag)
                    break;
            }
            continue;
        }
    }
}

std::string Handler::get_hdf5_file_path() {
    while (true) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
        bool flag = false;
        {
            std::lock_guard<std::mutex> lock(requestMutex);
            if (request == RECEIVED_HDF5_FILE_PATH)
                flag = true;
        }
        if (flag)
            break;
    }
    std::string returnVal;
    {
        std::lock_guard<std::mutex> lock(requestMutex);
        request = WAITING;
        returnVal = hdf5_file_path;
    }
    return returnVal;
}

void Handler::wait_for_done() {
    while (true) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
        bool flag = false;
        {
            std::lock_guard<std::mutex> lock(requestMutex);
            if (request == RECEIVED_DONE)
                flag = true;
        }
        if (flag)
            break;
    }
    {
        std::lock_guard<std::mutex> lock(requestMutex);
        request = WAITING;
    }
    return;
};

void Handler::send_done() {
    server.send("done");
    {
        std::lock_guard<std::mutex> lock(processingMutex);
        processing = false;
    }
}
