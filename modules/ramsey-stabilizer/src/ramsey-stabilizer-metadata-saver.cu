#include "ramsey-stabilizer-metadata-saver.h"

RamseyStabilizerMetadataSaver::RamseyStabilizerMetadataSaver(YAML::Node configs) {
    this->configs = std::move(configs);
    image_folder = this->configs["image_folder_name"].as<std::string>();
}

void ShotInformation::write_to(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&error_signal), sizeof(double));
    out.write(reinterpret_cast<const char*>(&frequency), sizeof(double));
    out.write(reinterpret_cast<const char*>(&new_frequency), sizeof(double));
    out.write(reinterpret_cast<const char*>(&frequency_moving_average), sizeof(double));
    out.write(reinterpret_cast<const char*>(&timestamp), sizeof(int64_t));
}

void RamseyStabilizerMetadataSaver::start() {
    this->thread_killed.store(false);
    this->thread_worker = std::make_unique<std::thread>(&RamseyStabilizerMetadataSaver::worker, this);
}

void RamseyStabilizerMetadataSaver::stop() {
    this->thread_killed.store(true);
    if (this->thread_worker != nullptr && this->thread_worker->joinable()) {
        this->thread_worker->join();
    }
}

void RamseyStabilizerMetadataSaver::worker() {
    ShotInformation new_info;
    auto delay_time_us = this->configs["delay_time_us"].as<size_t>();

    while (!this->thread_killed.load()) {
        if (this->queue.size() > 0) { // Intentionally not locked for the size check. We assume other threads cannot remove from the queue.
            {
                std::lock_guard<std::mutex> lock(this->mtx);
                new_info = this->queue.front();
                this->queue.pop_front();
            }

            try {
                this->save_to_file(new_info);
            } catch (const std::exception& e) {
                ERROR << "Exception in metadata saver: " << e.what() << std::endl;
            }
        }

        std::this_thread::sleep_for(std::chrono::microseconds(delay_time_us));
    }
}

void RamseyStabilizerMetadataSaver::add_to_queue(ShotInformation shot_information) {
    std::lock_guard<std::mutex> lock(this->mtx);
    this->queue.emplace_back(std::move(shot_information));
}

void RamseyStabilizerMetadataSaver::save_to_file(const ShotInformation& s) {
    boost::filesystem::path saving_dir = LabscriptAddressUtils::get_images_folder_name(s.shot_address, this->image_folder);
    auto target_address = saving_dir / boost::filesystem::path("ramsey-stabilizer.bin");

    std::ofstream fout(target_address.string(), std::ios::binary);
    if (!fout) {
        throw std::runtime_error("Could not open file for writing metadata");
    }

    s.write_to(fout);
    fout.close();
}

RamseyStabilizerMetadataSaver::~RamseyStabilizerMetadataSaver() {
    this->stop();
}