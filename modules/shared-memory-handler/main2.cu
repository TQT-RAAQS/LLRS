#include "shared-memory-handler.h"

int main() {
    auto smh = SharedMemoryHandler("default.yml");
    smh.open_connection();
    std::this_thread::sleep_for(std::chrono::seconds(50));
    smh.close_connection();

    return 0;
}