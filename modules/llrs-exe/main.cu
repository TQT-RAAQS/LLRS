#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>
#include "llrs-lib/Settings.h"
#include "Setup.h"
#include "llrs-lib/PreProc.h"
#include "JsonWrapper.h"
#include "Collector.h"
#include "Solver.h"
#include "WaveformRepo.h"
#include "WaveformTable.h"
#include "Sequence.h"
#include "llrs-lib/include/llrs.h"
#include "awg.hpp"
#include "trigger-detector.hpp"
#include <memory>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>


const int     llrs_idle_seg             = 1;
const int     llrs_idle_step            = 1;


void delay( float timeout ){
    
    auto startTime = std::chrono::high_resolution_clock::now();
    auto targetDuration = std::chrono::duration<float, std::micro>(timeout);

    while( true ){
        
        if( timeout != -1 && std::chrono::high_resolution_clock::now() - startTime >= targetDuration ){
            break;
        }

    }

}

void executeLLRS(LLRS<AWG>* l, std::shared_ptr<AWG> awg){
    
    int current_step = awg->get_current_step();

    l->execute();

    l->reset();
    std::cout << "Done LLRS::Execute" << std::endl;

    assert( current_step == llrs_idle_step );
    awg->seqmem_update( llrs_idle_step, llrs_idle_seg, 1, 0, SPCSEQ_ENDLOOPALWAYS );// this is slow
    delay(WAVEFORM_DUR * WF_PER_SEG * 1e6);

    current_step = awg->get_current_step();
    assert( current_step == 0 );

    // Ensure LLRS Idle is pointing to itself // move this into LLRS reset
    awg->seqmem_update(llrs_idle_step, llrs_idle_seg, 1, llrs_idle_step, SPCSEQ_ENDLOOPALWAYS);
    delay(WAVEFORM_DUR * WF_PER_SEG * 1e6);

}

void poll(LLRS<AWG>* l, std::shared_ptr<AWG> awg){

    int current_step;

    while(true){

        current_step = awg->get_current_step();

        if(current_step == llrs_idle_seg){
            executeLLRS(l, awg);
        }

    }

}

int main( int argc, char *argv[] ){


    // Read problem statement
    std::string problem_id;
    std::string problem_config;
    bool flag;

    if (argc > 3) {
        problem_config = std::string(argv[1]); // = "21_atoms_problem";
        problem_id = std::string(argv[2]); 
        flag = std::string(argv[3]) == "true";

    } else {
        ERROR << " No input was provided" << std::endl; 
        return LLRS_ERR;
    }


    std::shared_ptr<AWG> awg {std::make_shared<AWG>()};
    LLRS<AWG> *l = new LLRS<AWG>{awg}; 
    int16 *pnData = nullptr;
    int qwBufferSize = awg->allocate_transfer_buffer(awg->get_samples_per_segment(), pnData);
    awg->fill_transfer_buffer(pnData, awg->get_samples_per_segment(), 0);
    awg->init_and_load_all(pnData, awg->get_samples_per_segment());
    vFreeMemPageAligned(pnData, qwBufferSize);

    l->setup(problem_config, llrs_idle_seg, llrs_idle_step, problem_id);

    if(flag == true){
        l->get_1d_static_wfm( pnData, WF_PER_SEG );
    }

    // Load Segment 0 and 1
    awg->init_and_load_range( pnData, awg->get_samples_per_segment(), 0, 1 );
    
    // Segment 0 transitions to segment 1 in sequence memory
    awg->seqmem_update( 0, 0, 1, 1, SPCSEQ_ENDLOOPONTRIG );

    // Ensure there is enough time for the first idle segment's pointer to update
    delay(WAVEFORM_DUR * WF_PER_SEG * 1e6); 


    awg->start_stream();
    awg->print_awg_error();
    assert(awg->get_current_step() == 0);

    poll(l, awg);

    
    return 0;
}