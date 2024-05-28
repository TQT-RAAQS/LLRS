#include "server.hpp"
using json = nlohmann::json;

// Helper
void replaceStr(std::string &requestStr, std::string toReplace,
                std::string targetStr);

/**
 * @brief The constructor of the server class.
 *
 * Initialize a zmq context and tcp socket, and bind the socket to a
 * port.
 */
Server::Server() {
    // initialize the zmq context with a single IO thread
    context = zmq::context_t(1);
    socket = zmq::socket_t(context, zmq::socket_type::rep);
    socket.bind("tcp://*:5555");

    socket.set(zmq::sockopt::rcvtimeo, listen_timeout);
    socket.setsockopt(ZMQ_RCVTIMEO, listen_timeout);
}

void Server::set_listen_timeout(int timeout) {
    listen_timeout = timeout;
    socket.set(zmq::sockopt::rcvtimeo, listen_timeout);
    socket.setsockopt(ZMQ_RCVTIMEO, listen_timeout);
}
    
/**
 * @brief The destructor of the server class.
 *
 * Free any dynamically allocated memory segments
 */
Server::~Server() {
    socket.close();
    context.close();
}

bool Server::send(const std::string &string) {
    zmq::message_t message(string.size());
    memcpy(message.data(), string.data(), string.size());
    bool rc = socket.send(message);
    return rc;
}

/**
 * @brief Member function that listens on the socket for a message.
 *
 * Receive all messages on the configured socket. Parse the message
 * payload and the server reacts accordingly. All requests are met with
 * some sort of reply.
 */
int Server::listen(std::string &requestStr) {
    zmq::message_t request;
    zmq::recv_result_t result;

    try {
        result = socket.recv(request);
        if (!result) {
            std::cerr << "Receive failed." << std::endl;
            return 1;
        }
        requestStr =
            std::string(static_cast<char *>(request.data()), request.size());
    } catch (const zmq::error_t &e) {
        if (e.num() == EAGAIN) {
            std::cerr << "Receive timed out" << std::endl;
        } else {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        return 1;
    }
    return 0;
}


void Server::setMetadataAddress(std::string &requestStr1) {
    replaceStr(requestStr1, "\\", "/");
    replaceStr(requestStr1, "Z:", "/home/tqtraaqs1/Z");
    replaceStr(requestStr1, "labscript_shot_outputs", "llrs_data");
    replaceStr(requestStr1, ".h5", "/metadata.json");
    metadata_file_path = requestStr1;
}

int Server::llcs_handler() {

    int transition = -1;
    std::string requestStr;
    listen(requestStr);
    if (requestStr == "hello") {
        send("hello");
        transition = 4;
    } else if (requestStr == "abort") {
        send("done");
        transition = 3;
    } else if (requestStr == "psf") {
        std::string str = (PSF_CONFIG_PATH) + "default";
        const char *psf_translator = str.c_str();
        const char *command = "python3 ";
        char fullCommand[256];
        snprintf(fullCommand, sizeof(fullCommand), "%s%s", command,
                 psf_translator);
        int result = system(fullCommand);

        send("done");
        return result == 0? 4 : -1;
    } else if (requestStr == "done") { // The experimental shot is done
        send("ok");
        listen(requestStr);
        send("done");
        transition = 3;
    } else if (requestStr.substr(requestStr.length() - 3, 3) == ".h5") {
        std::string tempo = requestStr;
        setMetadataAddress(tempo);
        config_file_path = adjust_address(requestStr);
        send("ok");
        listen(requestStr); 
        send("done");
        transition = 2;
    } else if (requestStr == "SEND_DATA") {
        send(metadata_file_path);
        transition = 1;
    } else if (requestStr == "RESET") {
        send(metadata_file_path);
        transition = 5;
    } else if (requestStr.at(0) == '{') {
        json json_data = json::parse(requestStr);
        std::string llrs_reset_filepath = LLRS_RESET_PATH;
        std::ofstream json_file(llrs_reset_filepath);
        if (json_file.is_open()) {
            json_file << std::setw(4) << json_data;
            json_file.close();
            send("LLRS reconfig saved to " + llrs_reset_filepath);
            transition = 6;
        } else {
            send("Error opening file: " + llrs_reset_filepath);
            transition = -1;
        }
    } else if (requestStr == "control") {
        send("FSM taking control of AWG");
        transition = 7;
    } else {
        std::cerr << "Invalid message type received: " << requestStr
                  << std::endl;
        transition = -1;
    }

    return transition;
}

void replaceStr(std::string &requestStr, std::string toReplace,
                std::string targetStr) {
    size_t pos = requestStr.find(toReplace);

    // Check if double slashes are found
    while (pos != std::string::npos) {
        // Replace double slashes with a single slash
        requestStr.replace(pos, toReplace.length(), targetStr);

        // Search for the next occurrence
        pos = requestStr.find(toReplace, pos + 1);
    }
}

std::string adjust_address(std::string filename) {
    std::string adjusted_filename = "";
    int start_index = 0;
    if (filename.substr(0, 3) == "Z:\\") {
        start_index = 3;
        adjusted_filename = "/home/tqtraaqs1/Z/";
    }

    for (int i = start_index; i < filename.length(); i++) {
        adjusted_filename += (filename[i] == '\\') ? '/' : filename[i];
    }

    return adjusted_filename;
}
