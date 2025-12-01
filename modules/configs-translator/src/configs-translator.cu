#include "configs-translator.h"

ConfigsTranslator& ConfigsTranslator::instance() {
    static ConfigsTranslator obj;
    return obj;
}

ConfigsTranslator::ConfigsTranslator() {
    auto command = std::string("python ") + PSF_READER_SCRIPT;
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

void ConfigsTranslator::translate() {
    std::remove(PSF_TRANSLATION_FILE.c_str());
    std::remove(CONFIGS_TRANSLATION_READY_FILE.c_str());

    fprintf(this->translator_pipe, "reload\n");
    fflush(this->translator_pipe);

    INFO << "Trying to regenerate the translated config files...\n";
    do {
        if (fs::exists(CONFIGS_TRANSLATION_READY_FILE)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    } while (true);
    INFO << "Translated config files generated.\n";
}