/**
 * @brief   Server class that implements a ZMQ server
 * @date    Oct 2023
 */

#include "server.hpp"
using json = nlohmann::json;

/**
 * @brief The constructor of the server class.
 *
 * Initialize a zmq context and tcp socket, and bind the socket to a
 * port.
 */
Server::Server() {
    std::cout << "Server:: constructor" << std::endl;

    // initialize the zmq context with a single IO thread
    context = zmq::context_t(1);
    socket = zmq::socket_t(context, zmq::socket_type::rep);

    socket.bind("tcp://*:5555");

    // set a 50 second timeout on socket.recv() FS: not sure if this is
    // necessary
    socket.set(zmq::sockopt::rcvtimeo, 50000);
}

/**
 * @brief The destructor of the server class.
 *
 * Free any dynamically allocated memory segments
 */
Server::~Server() {
    std::cout << "Server:: destructor" << std::endl;

    // Close the socket
    socket.close();

    // Terminate the ZeroMQ context
    context.close();
}

bool Server::send(const std::string &string) {
    zmq::message_t message(string.size());
    std::memcpy(message.data(), string.data(), string.size());
    bool rc = socket.send(message);
    return (rc);
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

void Server::getMetadataAddress(std::string &requestStr1) {
    replaceStr(requestStr1, "\\", "/");
    replaceStr(requestStr1, "Z:", "/home/tqtraaqs1/Z");
    replaceStr(requestStr1, "labscript_shot_outputs", "llrs_data");
    replaceStr(requestStr1, ".h5", "/metadata.json");
    std::cout << requestStr1 << std::endl;
    metadata_file_path = requestStr1;
}
std::string Server::get_metadata_file_path() { return metadata_file_path; }

std::string Server::adjust_address(std::string filename) {
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

std::string Server::get_config_file_path() { return config_file_path; }
/**
 * @brief Member function that listens on the socket for a message.
 *
 * Receive all messages on the configured socket. Parse the message
 * payload and the server reacts accordingly. All requests are met with
 * some sort of reply.
 */
int Server::listen() {
    std::cout << "Server:: Listening" << std::endl;

    // listen for five seconds 50
    zmq::message_t request;
    zmq::recv_result_t result;
    int transition = -1;

    try {

        socket.setsockopt(ZMQ_RCVTIMEO, 50000000);

        result = socket.recv(request);
        if (!result) {
            std::cerr << "Receive failed." << std::endl;
            return -1;
        }

        std::string requestStr(static_cast<char *>(request.data()),
                               request.size());

        if (requestStr == "hello") {
            std::string outputStr = "hello";
            zmq::message_t output(outputStr.size());
            memcpy(output.data(), outputStr.data(), outputStr.size());
            socket.send(output);
            transition = 4;
        } else if (requestStr == "abort") {
            std::string outputStr = "done";
            zmq::message_t output(outputStr.size());
            memcpy(output.data(), outputStr.data(), outputStr.size());
            socket.send(output);
            transition = 3;
        } else if (requestStr == "psf") {
            std::string str = (PSF_CONFIG_PATH) + "default";
            const char *psf_translator = str.c_str();
            const char *command = "python3 ";

            char fullCommand[256];
            snprintf(fullCommand, sizeof(fullCommand), "%s%s", command,
                     psf_translator);

            int result = system(fullCommand);

            std::string outputStr = "done";
            zmq::message_t output(outputStr.size());
            memcpy(output.data(), outputStr.data(), outputStr.size());
            socket.send(output);
            if (result == 0) {
                // The Python script was executed successfully
                return 4;
            } else {
                // There was an error running the Python script
                return -1;
            }
        } else if (requestStr == "done") { // The experimental shot is done
            std::string statusMsg = "ok";
            std::cout << statusMsg << std::endl;
            zmq::message_t status(statusMsg.size());
            memcpy(status.data(), statusMsg.data(), statusMsg.size());
            socket.send(status);
            socket.recv(request);
            std::string outputStr = "done";
            zmq::message_t output(outputStr.size());
            memcpy(output.data(), outputStr.data(), outputStr.size());
            socket.send(output);
            transition = 3;
        } else if (requestStr.substr(requestStr.length() - 3, 3) == ".h5") {
            std::string statusMsg = "ok";

            std::string tempo = requestStr;
            getMetadataAddress(tempo);
            std::cout << tempo << std::endl;
            config_file_path = adjust_address(requestStr);
            std::cout << statusMsg << std::endl;
            zmq::message_t status(statusMsg.size());
            memcpy(status.data(), statusMsg.data(), statusMsg.size());
            socket.send(status);

            socket.recv(request);
            statusMsg = "done";
            std::cout << statusMsg << std::endl;
            zmq::message_t status1(statusMsg.size());
            memcpy(status1.data(), statusMsg.data(), statusMsg.size());
            socket.send(status1);
            std::cout << "Got it" << std::endl;
            transition = 2;
        } else if (requestStr == "SEND_DATA") {
            std::string statusMsg = metadata_file_path;
            zmq::message_t status(statusMsg.size());
            memcpy(status.data(), statusMsg.data(), statusMsg.size());
            socket.send(status);
            transition = 1;
        } else if (requestStr == "RESET") {
            std::string statusMsg = metadata_file_path;
            zmq::message_t status(statusMsg.size());
            memcpy(status.data(), statusMsg.data(), statusMsg.size());
            socket.send(status);
            transition = 5;
        } else if (requestStr.at(0) == '{') {

            json json_data = json::parse(requestStr);

            std::string llrs_reset_filepath = LLRS_RESET_PATH;

            std::ofstream json_file(llrs_reset_filepath);
            if (json_file.is_open()) {
                json_file << std::setw(4) << json_data;
                json_file.close();
                std::string statusMsg =
                    "LLRS reconfig saved to " + llrs_reset_filepath;
                zmq::message_t status(statusMsg.size());
                memcpy(status.data(), statusMsg.data(), statusMsg.size());
                socket.send(status);
                transition = 6;
            } else {
                std::string statusMsg =
                    "Error opening file: " + llrs_reset_filepath;
                zmq::message_t status(statusMsg.size());
                memcpy(status.data(), statusMsg.data(), statusMsg.size());
                socket.send(status);
                transition = -1;
            }

        } else if (requestStr == "control") {
            std::string statusMsg = "FSM taking control of AWG";
            zmq::message_t status(statusMsg.size());
            memcpy(status.data(), statusMsg.data(), statusMsg.size());
            socket.send(status);
            transition = 7;
        } else {
            std::cerr << "Invalid message type received: " << requestStr
                      << std::endl;
            transition = -1;
        }
    } catch (const zmq::error_t &e) {
        if (e.num() == EAGAIN) {
            // Timeout occurred
            std::cout << "Receive timed out" << std::endl;
        } else {
            // Some other error
            std::cout << "Error: " << e.what() << std::endl;
        }
    }

    return transition;
}
