#include "trap-result-saver.h"

TrapResultSaver::TrapResultSaver(const std::string config) {
    this->configs = YAML::LoadFile(TRAP_RESULT_SAVER(config));
    this->image_folder_name = this->configs["image_folder_name"].as<std::string>();
    this->setup_memory_handler();
    this->setup_semaphore();
}

void TrapResultSaver::setup_semaphore() {
    sem_init(this->saving_semaphore.get(), 0, 0);
}

void TrapResultSaver::setup_memory_handler() {
    auto config_name = this->configs["smh_config"].as<std::string>();
    this->memory_handler = std::make_unique<SharedMemoryHandler>(config_name);
    this->memory_handler->open_connection();
}

void TrapResultSaver::start() {
    this->thread_killed.store(false);
    this->saver_thread = std::make_unique<std::thread>(&TrapResultSaver::saver_worker, this);
    this->data_retriever_thread = std::make_unique<std::thread>(&TrapResultSaver::retriever_worker, this);
}

void TrapResultSaver::saver_worker() {
    auto saver_timeout_s = this->configs["saver_timeout_s"].as<uint8_t>();

    while (!this->thread_killed.load()) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += saver_timeout_s;

        auto ret = sem_timedwait(this->saving_semaphore.get(), &ts);
        if (ret == -1) {
            if (errno == ETIMEDOUT) {
                if (this->thread_killed.load()) break;
                continue;
            } else {
                throw std::system_error(errno, std::generic_category(), "sem_timedwait failed");
            }
        }

        ShotTrapResult results_to_save;
        {
            std::lock_guard<std::mutex> lock(this->trap_results_mutex);
            if (this->trap_results.empty()) {
                continue;
            }
            results_to_save = std::move(this->trap_results.front());
            this->trap_results.pop();
        }

        this->save_to_file(results_to_save);
    }
}

void TrapResultSaver::save_to_file(ShotTrapResult& shot_trap_result) {
    auto& dir_address = std::get<0>(shot_trap_result);
    auto& results     = std::get<1>(shot_trap_result);

    auto file_address = (boost::filesystem::path(dir_address) / "traps.bin").string();

    std::ofstream ofs(file_address, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + file_address);
    }

    uint64_t num_images = results.size();
    ofs.write(reinterpret_cast<const char*>(&num_images), sizeof(num_images));

    for (auto& trap : results) {
        auto& fls_counts = std::get<0>(trap);
        auto& occupancy  = std::get<1>(trap);

        uint64_t trap_size = fls_counts.size();
        ofs.write(reinterpret_cast<const char*>(&trap_size), sizeof(trap_size));

        ofs.write(reinterpret_cast<const char*>(fls_counts.data()),
                  fls_counts.size() * sizeof(double_t));

        ofs.write(reinterpret_cast<const char*>(occupancy.data()),
                  occupancy.size() * sizeof(uint8_t));
    }

    ofs.close();
}

void TrapResultSaver::retriever_worker() {
    int16_t processed_image_count = SHOT_NOT_BEGUN_YET;
    auto current_image_count = this->memory_handler->get_image_count();
    std::string saving_address = "";
    if (current_image_count != 0) {
        throw std::runtime_error("The expected number of images in the shared memory is initially 0. This is unexpected.");
    }

    while (!this->thread_killed.load()) {
        this->memory_handler->wait_for_update(); // Wait for an update trigger from the master

        current_image_count = this->memory_handler->get_image_count();

        INFO << "Current image count: " << std::to_string(current_image_count) << ", processed image count: " << std::to_string(processed_image_count) << std::endl;

        if (processed_image_count == SHOT_NOT_BEGUN_YET && current_image_count == 0) { // The shot has begun
            auto shot_address = this->memory_handler->get_shot_address();
            this->memory_handler->signal_done();
            
            saving_address = LabscriptAddressUtils::get_images_folder_name(shot_address, this->image_folder_name);
            processed_image_count = 0;

            {
                std::lock_guard<std::mutex> lock(this->trap_results_mutex);
                this->trap_results.emplace(saving_address, std::vector<ImageTrapResult>{}); // Add empty image trap results to the queue
            }

        } else if (processed_image_count != SHOT_NOT_BEGUN_YET && current_image_count > 0 && current_image_count > processed_image_count) { // New image has arrived

            this->add_data_to_queue(processed_image_count, saving_address);
            processed_image_count++;

        } else if (processed_image_count == current_image_count) { // The shot is done
            INFO << processed_image_count << " images processed and to be saved in " << saving_address << std::endl;
            sem_post(this->saving_semaphore.get());
            this->memory_handler->signal_done();

            processed_image_count = SHOT_NOT_BEGUN_YET;

        } else {
            throw std::runtime_error("Unexpected case in the memory manager of the trap result saver shared memory handler. This is most likely a bug. Current image count: " + \
                std::to_string(current_image_count) + ", processed image count: " + std::to_string(processed_image_count) + ".");
        }
    }
}

void TrapResultSaver::add_data_to_queue(size_t image_index, std::string save_directory) {
    auto fls_counts = this->memory_handler->get_trap_fluorescence(image_index);
    auto occupancy = this->memory_handler->get_trap_occupancy(image_index);
    
    std::get<1>(this->trap_results.back()).emplace_back(std::move(fls_counts), std::move(occupancy));
}

void TrapResultSaver::stop() {
    this->thread_killed.store(true);
    if (this->data_retriever_thread->joinable()) {
        this->data_retriever_thread->join();
    }
    if (this->saver_thread->joinable()) {
        this->saver_thread->join();
    }
    this->memory_handler->close_connection();
}

TrapResultSaver::~TrapResultSaver() {
    this->stop();
}