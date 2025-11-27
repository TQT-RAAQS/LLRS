#include "master-shared-memory-handler.h"
#include <csignal>

int MasterSharedMemoryHandler::open_connection() {
    auto output = SharedMemoryHandler::open_connection();

    // Only the master is allowed to create the memory environment.
    if (output == 2) {
        return output;
    }

    this->kill_subscribers();

    this->close_connection();
    this->clear_memory();
    output = SharedMemoryHandler::open_connection();
    if (output != 2) {
        throw std::runtime_error("Unexpected failure to open connection as a master memory handler.");
    }
    return output;
}

void MasterSharedMemoryHandler::kill_subscribers() {
    auto subscriber_count = this->shared_memory->subscription_count;
    pid_t p;

    for (size_t i = 0; i < subscriber_count; ++i) {
        p = this->shared_memory->subscriber_pids.at(i);
        if (p != this->pid && kill(p, 0) == 0) { // check if alive
            kill(p, SIGKILL);
        }
    }
}