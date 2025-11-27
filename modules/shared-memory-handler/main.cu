#include "master-shared-memory-handler.h"

int main() {
    auto smh = MasterSharedMemoryHandler("default.yml");
    smh.open_connection();

    std::this_thread::sleep_for(std::chrono::seconds(10));

    smh.close_connection();

    return 0;
}