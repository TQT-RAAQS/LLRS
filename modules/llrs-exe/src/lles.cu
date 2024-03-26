#include "lles.h"

const int     llrs_idle_seg             = 1;
const int     llrs_idle_step            = 1;


template<typename AWG_T>LLES<AWG_T>::LLES( TriggerDetector<AWG_T> *td ): trigger_detector(td){
    auto awg = trigger_detector->getAWG();
    l = new LLRS<AWG_T>{awg};
};


template<typename AWG_T>LLES<AWG_T>::~LLES () {
    std::cout << "FSM:: destructor" << std::endl;

}

template<typename AWG_T>SegFlag LLES<AWG_T>::setFlag(){

}

template<typename AWG_T>void LLES<AWG_T>::executeLLRS(){
    
    int current_step = awg->get_current_step();

    l->execute();

    l->reset();
    std::cout << "Done LLRS::Execute" << std::endl;

    assert( current_step == llrs_idle_step );
    awg->seqmem_update( llrs_idle_step, llrs_idle_seg, 1, 0, SPCSEQ_ENDLOOPALWAYS );// this is slow
    trigger_detector->busyWait(WAVEFORM_DUR * WF_PER_SEG * 1e6);

    current_step = awg->get_current_step();
    assert( current_step == 0 );

    // Ensure LLRS Idle is pointing to itself // move this into LLRS reset
    awg->seqmem_update(llrs_idle_step, llrs_idle_seg, 1, llrs_idle_step, SPCSEQ_ENDLOOPALWAYS);
    trigger_detector->busyWait(WAVEFORM_DUR * WF_PER_SEG * 1e6);

}


template<typename AWG_T>void LLES<AWG_T>::pollSeg1(){
    int current_step;

    while(true){

        current_step = awg->get_current_step();

        if(current_step == llrs_idle_seg){
            executeLLRS();
        }

    }

}

template<typename AWG_T>void LLES<AWG_T>::runLLES(){




}

template class LLES<AWG>;
