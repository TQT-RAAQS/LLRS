#ifndef CONFIGS_TRANSLATOR_H_
#define CONFIGS_TRANSLATOR_H_

#include <fstream>
#include <thread>
#include "llrs-lib/PreProc.h"

class ConfigsTranslator {

    FILE* translator_pipe;
    
    ConfigsTranslator();
    
public:
    static ConfigsTranslator& instance();
    ~ConfigsTranslator();

    void translate_psf();

    ConfigsTranslator(const ConfigsTranslator&) = delete;
    ConfigsTranslator& operator=(const ConfigsTranslator&) = delete;
};

#endif