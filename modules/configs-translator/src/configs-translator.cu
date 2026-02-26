#include "configs-translator.h"

ConfigsTranslator& ConfigsTranslator::instance() {
    static ConfigsTranslator obj;
    return obj;
}

ConfigsTranslator::ConfigsTranslator() {
    auto command = std::string("python ") + CONFIGS_TRANSLATOR_SCRIPT;
    this->translator_pipe = popen(command.c_str(), "w");
    if (!this->translator_pipe) {
        ERROR << "Could not open the pipe to the psf reader script.\n";
        throw std::runtime_error("Could not open the psf reader.");
    }
}

ConfigsTranslator::~ConfigsTranslator() {
    fprintf(this->translator_pipe, "quit\n");
    fflush(this->translator_pipe);
    pclose(this->translator_pipe);
}

void ConfigsTranslator::translate_psf() {
    std::remove(PSF_TRANSLATION_FILE.c_str());
    std::remove(CONFIGS_PSF_TRANSLATION_READY_FILE.c_str());
    std::remove(TRAPS_ORDERS_TRANSLATION_FILE.c_str());

    fprintf(this->translator_pipe, "reload_psf\n");
    fflush(this->translator_pipe);

    INFO << "Trying to regenerate the translated config files...\n";
    do {
        if (fs::exists(CONFIGS_PSF_TRANSLATION_READY_FILE)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    } while (true);
    INFO << "Translated config files generated.\n";
}

void ConfigsTranslator::translate_iqmixer() {
    std::remove(IQMIXER_TRANSLATION_FILE.c_str());
    std::remove(IQMIXER_TRANSLATION_READY_FILE.c_str());

    fprintf(this->translator_pipe, "reload_iqmixer\n");
    fflush(this->translator_pipe);

    INFO << "Trying to regenerate the translated config files...\n";
    do {
        if (fs::exists(IQMIXER_TRANSLATION_READY_FILE)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    } while (true);
    INFO << "Translated iqmixer config files generated.\n";
}

void ConfigsTranslator::translate_linear_controller_configs() {
    std::remove(LINEAR_CONTROLLER_TRANSLATION_FILE.c_str());
    std::remove(LINEAR_CONTROLLER_TRANSLATION_READY_FILE.c_str());

    fprintf(this->translator_pipe, "reload_linear_controller_configs\n");
    fflush(this->translator_pipe);

    INFO << "Trying to regenerate the translated linear controller config files...\n";
    do {
        if (fs::exists(LINEAR_CONTROLLER_TRANSLATION_READY_FILE)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    } while (true);
    INFO << "Translated linear controller config files generated.\n";
}