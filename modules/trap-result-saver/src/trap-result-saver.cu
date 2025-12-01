#include "trap-result-saver.h"

TrapResultSaver::TrapResultSaver(const std::string config) {
    this->configs = YAML::LoadFile(TRAP_RESULT_SAVER(config));
    this->image_folder_name = this->configs["image_folder_name"].as<std::string>();
    this->setup_memory_handler();
}

void TrapResultSaver::setup_memory_handler() {
    auto config_name = this->configs["smh_config"].as<std::string>();
    this->memory_handler = std::make_unique<SharedMemoryHandler>(config_name);
    this->memory_handler->open_connection();
}

void TrapResultSaver::start() {
    this->thread_killed.store(false);
    this->saver_thread = std::make_unique<std::thread>(&TrapResultSaver::saver_worker, this);
}

void TrapResultSaver::saver_worker() {
    int16_t processed_image_count = SHOT_NOT_BEGUN;
    auto current_image_count = this->memory_handler->get_image_count();
    std::string saving_address = "";
    if (current_image_count != 0) {
        throw std::runtime_error("The expected number of images in the shared memory is initially 0. This is unexpected.");
    }

    while (!this->thread_killed.load()) {
        this->memory_handler->wait_for_update(); // Wait for an update trigger from the master

        current_image_count = this->memory_handler->get_image_count();

        INFO << "Current image count: " << std::to_string(current_image_count) << ", processed image count: " << std::to_string(processed_image_count) << std::endl;

        if (processed_image_count == SHOT_NOT_BEGUN && current_image_count == 0) { // The shot has begun
            auto shot_address = this->memory_handler->get_shot_address();
            this->memory_handler->signal_done();
            
            saving_address = LabscriptAddressUtils::get_images_folder_name(shot_address, this->image_folder_name);
            processed_image_count = 0;

        } else if (processed_image_count != SHOT_NOT_BEGUN && current_image_count > 0 && current_image_count > processed_image_count) { // New image has arrived

            // process images
            processed_image_count++;

        } else if (processed_image_count == current_image_count) { // The shot is done
            INFO << processed_image_count << " images processed and to be saved in " << saving_address << std::endl;
            this->memory_handler->signal_done();

            processed_image_count = SHOT_NOT_BEGUN;

        } else {
            throw std::runtime_error("Unexpected case in the memory manager of the trap result saver shared memory handler. This is most likely a bug. Current image count: " + \
                std::to_string(current_image_count) + ", processed image count: " + std::to_string(processed_image_count) + ".");
        }
    }
}

void TrapResultSaver::stop() {
    this->thread_killed.store(true);
    if (this->saver_thread->joinable()) {
        this->saver_thread->join();
    }
    this->memory_handler->close_connection();
}

TrapResultSaver::~TrapResultSaver() {
    this->stop();
}