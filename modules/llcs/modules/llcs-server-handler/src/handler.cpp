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
    std::string requestStrNull;
    while (true) {
        server.listen(requestStr);
        if (requestStr == "hello") {
            server.send("hello");
            continue;
        } else if (requestStr == "done") {
            server.send("ok");
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
                std::this_thread::sleep_for(std::chrono::microseconds(1));
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
            server.listen(requestStrNull);
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
                std::this_thread::sleep_for(std::chrono::microseconds(1));
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
        } else if (requestStr == "psf") {
            {
                std::lock_guard<std::mutex> lock(requestMutex);
                request = RECEIVED_PSF_RESET_REQUEST;
            }
            {
                std::lock_guard<std::mutex> lock(processingMutex);
                processing = true;
            }
            while (true) {
                std::this_thread::sleep_for(std::chrono::microseconds(1));
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
        } else if (requestStr == "waveform") {
            {
                std::lock_guard<std::mutex> lock(requestMutex);
                request = RECEIVED_WAVEFORM_RESET_REQUEST;
            }
            {
                std::lock_guard<std::mutex> lock(processingMutex);
                processing = true;
            }
            while (true) {
                std::this_thread::sleep_for(std::chrono::microseconds(1));
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
        } else if (requestStr == "awg") {
            {
                std::lock_guard<std::mutex> lock(requestMutex);
                request = RECEIVED_AWG_RESET_REQUEST;
            }
            {
                std::lock_guard<std::mutex> lock(processingMutex);
                processing = true;
            }
            while (true) {
                std::this_thread::sleep_for(std::chrono::microseconds(1));
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
        } else if (requestStr.substr(requestStr.length() - 4, 4) == ".yml") {
            {
                std::lock_guard<std::mutex> lock(requestMutex);
                llrs_config_file = requestStr;
                request = RECEIVED_RESET_REQUEST;
            }
            {
                std::lock_guard<std::mutex> lock(processingMutex);
                processing = true;
            }
            while (true) {
                std::this_thread::sleep_for(std::chrono::microseconds(1));
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

uint Handler::get_request() {
    uint req;
    while (true) {
        {
            std::lock_guard<std::mutex> lock(requestMutex);
            req = request;
            request = WAITING;
        }
        if (req != WAITING) {
            std::cout << "Request received:" << req << std::endl;
            return req;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
}

std::string Handler::get_hdf5_file_path() {
    std::string hdf5_file_path;
    {
        std::lock_guard<std::mutex> lock(requestMutex);
        hdf5_file_path = this->hdf5_file_path;
    }
    return hdf5_file_path;
}

std::string Handler::get_llrs_config_file() {
    std::string config_file;
    {
        std::lock_guard<std::mutex> lock(requestMutex);
        config_file = llrs_config_file;
    }
    return config_file;
}

void Handler::wait_for_done() {
    while (true) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
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

void Handler::send_200() {
    server.send("200");
    {
        std::lock_guard<std::mutex> lock(processingMutex);
        processing = false;
    }
}
