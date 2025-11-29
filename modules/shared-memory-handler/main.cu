#include "master-shared-memory-handler-server.h"
#include <limits.h>

int main(int argc, char* args[]) {
    char hostname[HOST_NAME_MAX + 1];
    gethostname(hostname, sizeof(hostname));
    
    std::string config_index = ( argc == 1 ? "" : "-" + std::string(args[1]) );
    
    std::string config_name = std::string(hostname) + config_index;

    INFO << "Config name: " + config_name + "\n";
    INFO << "Trying to open the shared memory handler server...\n";

    MasterSharedMemoryHandlerServer server(config_name + std::string(".yml"));
    
    server.start();
    INFO << "Shared memory handler server has initialized.\n";

    server.wait_until_server_closed();

    return 0;
}