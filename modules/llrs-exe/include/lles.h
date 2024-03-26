#ifndef LLES_H
#define LLES_H

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
using json = nlohmann::json;

enum SegFlag {
    STATIC,
    NOTHING
};

template<typename AWG_T> class LLES{


    TriggerDetector<AWG_T> *trigger_detector;
    LLRS<AWG_T> *l; 
    int num_exp_sequence;
    int16 *pnData;
    int qwBufferSize;

    SegFlag setFlag();

    void pollSeg1();
    void executeLLRS();
    void runLLES();

public:
    LLES( TriggerDetector<AWG_T> *td );
    ~LLES();
}


