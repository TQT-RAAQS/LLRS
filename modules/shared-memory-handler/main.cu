#include "master-shared-memory-handler.h"

int main() {
    auto smh = MasterSharedMemoryHandler("default.yml");
    smh.open_connection();

    smh.close_connection();

    return 0;
}