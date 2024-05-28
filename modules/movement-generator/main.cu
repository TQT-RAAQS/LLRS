#include "awg.hpp"
#include "llrs-lib/PreProc.h"
#include "handler.hpp"
#include "shot-file.h"
#include "synthesiser.h"
#include <string>


int main() {
    AWG awg{};
    std::string hdf_address{};
    Handler server_handler{};
    server_handler.start_listening();

    while (!server_handler.get_abort()) {

        hdf_address = server_handler.get_hdf5_file_path(); 

        ShotFile shotfile(hdf_address);
        MovementsConfig movementsConfig(shotfile);
        Synthesiser synthesiser{COEF_X_PATH("21_traps.csv"),
                                COEF_Y_PATH("21_traps.csv"), movementsConfig};
        synthesiser.synthesise_and_upload(awg);

        awg.start_stream();
        server_handler.send_done();

        server_handler.wait_for_done();
        awg.stop_card();
        server_handler.send_done();
        
    }

    server_handler.send_done();
}
