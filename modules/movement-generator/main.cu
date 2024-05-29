#include "awg.hpp"
#include "llrs-lib/PreProc.h"
#include "handler.hpp"
#include "shot-file.h"
#include "synthesiser.h"
#include <string>


int main() {
    std::string hdf_address{};
    Handler server_handler{};
    server_handler.start_listening();
    std::cout << "Server started" << std::endl;

    while (!server_handler.get_abort()) {

        std::cout << "wait for hdf" << std::endl;
        hdf_address = server_handler.get_hdf5_file_path(); 
        /*
    AWG awg{};
        ShotFile shotfile(hdf_address);
        MovementsConfig movementsConfig(shotfile);
        Synthesiser synthesiser{COEF_X_PATH("21_traps.csv"),
                                COEF_Y_PATH("21_traps.csv"), movementsConfig};
        synthesiser.synthesise_and_upload(awg);
        awg.start_stream();
        awg.stop_card();
        */
        std::cout << "Start Stream" << std::endl;
        server_handler.send_done();

        std::cout << "wait for done" << std::endl;
        server_handler.wait_for_done();

        std::cout << "sending done" << std::endl;
        server_handler.send_done();
        
    }

    server_handler.send_done();
}
