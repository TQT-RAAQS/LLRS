#include "master-shared-memory-handler-server.h"
#include <limits.h>

int main() {
    char hostname[HOST_NAME_MAX + 1];
    gethostname(hostname, sizeof(hostname));
    INFO << "Host: " + std::string(hostname) + "\n";
    INFO << "Trying to open the shared memory handler server...\n";

    MasterSharedMemoryHandlerServer server(std::string(hostname) + std::string(".yml"));
    
    server.start();
    INFO << "Shared memory handler server has initialized.\n";

    server.wait_until_server_closed();

    return 0;
}