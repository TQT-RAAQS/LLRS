#include "awg.hpp"
#include "llrs-lib/PreProc.h"
#include "handler.hpp"
#include "shot-file.h"
#include "synthesiser.h"
#include <string>
#include <signal.h>

void my_handler(int s) {
    printf("Caught signal %d\n", s);
    exit(1);
}

int main() {
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);


    std::string hdf_address{};
    Handler server_handler{};
    server_handler.start_listening();
    std::cout << "Server started" << std::endl;

    while (true) {

        AWG awg{};
        std::cout << "wait for hdf" << std::endl;
        hdf_address = server_handler.get_hdf5_file_path(); 
        
        ShotFile shotfile(hdf_address);
        MovementsConfig movementsConfig(shotfile);
        Synthesiser synthesiser{COEF_X_PATH("21_traps.csv"),
                                COEF_Y_PATH("21_traps.csv"), movementsConfig};
        synthesiser.synthesise_and_upload(awg);
        awg.start_stream();

        std::cout << "Start Stream" << std::endl;
        server_handler.send_done();

        std::cout << "wait for done" << std::endl;
        server_handler.wait_for_done();
        
        awg.stop_card();

        std::cout << "sending done" << std::endl;
        server_handler.send_done();
        
    }

}
