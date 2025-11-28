#include "master-shared-memory-handler.h"

int main() {
    auto smh = MasterSharedMemoryHandler("master-default.yml");
    smh.open_connection();

    // std::this_thread::sleep_for(std::chrono::seconds(8));

    // smh.close_connection();

    return 0;
}