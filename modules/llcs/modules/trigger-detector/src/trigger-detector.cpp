#include "trigger-detector.hpp"
/**
 * @brief Class that contains the necessary code to detect hardware trigger events
*/
template<typename AWG_T> TriggerDetector<AWG_T>::TriggerDetector(): awg{std::make_shared<AWG_T>()}, samples_per_td_segment{awg->get_samples_per_segment()} {}


template<typename AWG_T> void TriggerDetector<AWG_T>::generateSineWave( int16 *pnData, int samples, double sampleRate ){
    double frequency = sampleRate / samples * 10000;    // about 30MHz signal
    double amplitude = 8000;
    double twoPi = 2.0 * M_PI;

    int lSetChannels = awg->get_num_channels();
    int dwFactor = 1;
    int dwSegmentLenSample = dwFactor * samples;
    for( int i = 0; i < dwSegmentLenSample; i++ ){
        double t = static_cast<double>(i) / sampleRate;
        double angle = twoPi * frequency * t;
        double sineValue = sin(angle);
        // pnData[i] = sineValue * amplitude;
        // pnData[2*i] = sineValue * amplitude;
        // pnData[2*i + 1] = sineValue * amplitude;
        for( int lChannel = 0; lChannel < lSetChannels; ++lChannel ){
            pnData[i * lSetChannels + lChannel] = sineValue * amplitude;
        }

    }
}

template<typename AWG_T> int TriggerDetector<AWG_T>::setup( int16 *pnData ){
    
    // DATA MEMORY
    awg->init_and_load_range( pnData, samples_per_td_segment, 0, 1 );
    
    // SEQUENCE MEMORY
    awg->seqmem_update( 0, 0, 1, 1, SPCSEQ_ENDLOOPONTRIG );

    // Ensure there is enough time for the first idle segment's pointer to update
    busyWait(WAVEFORM_DUR * WF_PER_SEG * 1e6); 

    return SYS_OK;
}

template<typename AWG_T> int TriggerDetector<AWG_T>::busyWait( float timeout ){
    
    // std::cout << "TD Busy Waiting for " << timeout << " microseconds" << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    auto targetDuration = std::chrono::duration<float, std::micro>(timeout);

    while( true ){
        
        // Timeout loop break
        if( timeout != -1 && std::chrono::high_resolution_clock::now() - startTime >= targetDuration ){
            break;
        }
    }

    return SYS_OK;
}

template<typename AWG_T>int TriggerDetector<AWG_T>::resetDetectionSegments(){
    
    awg->seqmem_update( 0, 0, 1, 1, SPCSEQ_ENDLOOPONTRIG );

    return SYS_OK;
}

/**
 * @brief Function that detects a hardware trigger
 * 
 * If a timeout value is given, then that value is used, otherwise
 * it will listen for a hw trigger forever.
*/
template<typename AWG_T>int TriggerDetector<AWG_T>::detectTrigger( int timeout ){
    int current_seg = -1;
    int last_seg = -1;
    int counter = 0;

    spcm_dwGetParam_i32( awg->get_card(), SPC_SEQMODE_STATUS, &current_seg );
    last_seg = current_seg;

    // std::cout << "Starting Current Seg: " << current_seg << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();
    auto targetDuration = std::chrono::seconds(timeout);

    while( true ){
        // Timeout loop break
        if( timeout != -1 && std::chrono::high_resolution_clock::now() - startTime >= targetDuration ){
            break;
        }

        spcm_dwGetParam_i32( awg->get_card(), SPC_SEQMODE_STATUS, &current_seg );
        // std::cout << "Current Seg: " << current_seg << std::endl;

        if( (current_seg != last_seg) ){
            // trigger event occured
            std::cout << "TriggerDetector: trigger event occured - Current Seg: " << current_seg << std::endl;
            std::cout << "last_seg: " << last_seg << std::endl;
            counter++;
            last_seg = current_seg;
            
            return current_seg;
        }
        
    }
    
    return NO_HW_TRIG;

}

template<typename AWG_T> std::shared_ptr<AWG_T> &TriggerDetector<AWG_T>::getAWG(){
    return awg;
}

template class TriggerDetector<AWG>;

  