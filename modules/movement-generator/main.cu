#include "awg.hpp"
#include "handler.hpp"
#include "llrs-lib/PreProc.h"
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


	AWG awg{};
    std::string hdf_address{};
    Handler server_handler{};
    {
        AWG::TransferBuffer buffer = awg.allocate_transfer_buffer(
        static_cast<int>(10e-6 * awg.get_sample_rate()), false);
        awg.fill_transfer_buffer(buffer, static_cast<int>(10e-6 * awg.get_sample_rate()), 0);
    awg.init_and_load_range(
        *buffer, static_cast<int>(10e-6 * awg.get_sample_rate()), 0,
        1);
        awg.seqmem_update(0, 0, 1, 0, SPCSEQ_ENDLOOPALWAYS);
    }
    awg.start_stream();
    server_handler.start_listening();
    std::cout << "Server started" << std::endl;

    while (true) {
		std::cout << "Waiting for HDF5 address" << std::endl;
        hdf_address = server_handler.get_hdf5_file_path(); 
        ShotFile shotfile(hdf_address);
        MovementsConfig movementsConfig(shotfile);
        shotfile.close_file();

        Synthesiser synthesiser{COEF_X_PATH("21_traps.csv"),
                                COEF_Y_PATH("21_traps.csv"), movementsConfig};
        synthesiser.synthesise_and_upload(awg, 1);
        std::cout << "Start Stream" << std::endl;
        server_handler.send_done();

        std::cout << "wait for done" << std::endl;
        server_handler.wait_for_done();

        synthesiser.reset(awg, 1);

        std::cout << "sending done" << std::endl;
        server_handler.send_done();
    }

}
