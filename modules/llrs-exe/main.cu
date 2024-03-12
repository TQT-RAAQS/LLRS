#include "llrs.h"
#include <fstream>
#include <cstdlib>
#include <memory>

int main(int argc, char * argv[]){
    
    // Read problem statement
    std::string problem_id;
    std::string problem_config;
    if (argc > 1) {
        problem_config = std::string(argv[1]); 
        problem_id = argv>2?std::string(argv[2]): ""; 
    } else {
        ERROR << " No argument provided, reading from the default config file." << std::endl; 
        problem_config = "config.yml";
        return LLRS_ERR;
    }

    std::shared_ptr<AWG> awg {std::make_shared<AWG>()};
    LLRS<AWG> l {awg}; 
    int16 *pnData = nullptr;
    int qwBufferSize = awg->allocate_transfer_buffer(awg->get_samples_per_segment(), pnData);
    awg->fill_transfer_buffer(pnData, awg->get_samples_per_segment(), 0);
    awg->init_and_load_all(pnData, awg->get_samples_per_segment());
    vFreeMemPageAligned(pnData, qwBufferSize);

    l.setup(problem_config, 0, 0, problem_id);

    awg->start_stream();
    awg->print_awg_error();
    assert(awg->get_current_step() == 0);

    std::cout << "LLRS Setup Complete, please press any key to execure LLRS." << std::endl;
    std::cin.get();

    l.execute();
    l.reset();
}
