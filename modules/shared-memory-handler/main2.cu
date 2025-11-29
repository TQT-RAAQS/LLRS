#include "shared-memory-handler.h"

int main() {
    int tw = 100, th = 100;
    int n = tw * th;
    std::vector<double_t> f(n, 5.5);
    std::vector<uint8_t> a(n, 1);

    auto smh = SharedMemoryHandler("default.yml");
    smh.open_connection();
    
    smh.register_as_image_saver();
    
    smh.save_trap_array_information(tw, th, f, a);

    smh.close_connection();

    return 0;
}