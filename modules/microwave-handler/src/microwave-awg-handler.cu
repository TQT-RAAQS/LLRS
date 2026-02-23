
#include "microwave-awg-handler.h"

using namespace MicrowaveHandler;

MicrowaveHandler::MicrowaveAwgHandler::MicrowaveAwgHandler(const std::string& handler_config) {
    this->reload();

    auto config = YAML::LoadFile(MICROWAVE_AWG_HANDLER_CONFIG(handler_config));

    auto awg_config = config["awg_config"].as<std::string>();
    this->awg = AWG(awg_config);

    this->max_segment_count = config["max_segment_count"].as<int>();
    this->default_pause_segment_size = config["default_pause_segment_size"].as<int>();
    this->digital_offset_time = config["digital_offset_time"].as<double>();
    this->min_segment_size = this->awg.get_minimum_segment_size();
    this->segment_size_steps = this->awg.get_segment_size_steps();

    auto synthesizer_fast_flag = config["synthesizer_fast_interleaving_flag"].as<bool>(true);

    this->timer_worker_wait_time_ms = config["timer_worker_wait_time_ms"].as<int>();

    this->synthesizer.set_digital_offset_time(this->digital_offset_time);
    this->synthesizer.set_fast_interleaving_flag(synthesizer_fast_flag);

    if (!(this->default_pause_segment_size >= this->min_segment_size)) {
        throw std::runtime_error(
            "default_pause_segment_size (" + std::to_string(this->default_pause_segment_size) + 
            ") must be >= min_segment_size (" + std::to_string(this->min_segment_size) + ")"
        );
    }
    
    if (!(this->default_pause_segment_size % this->segment_size_steps == 0)) {
        throw std::runtime_error(
            "default_pause_segment_size (" + std::to_string(this->default_pause_segment_size) + 
            ") must be divisible by segment_size_steps (" + std::to_string(this->segment_size_steps) + ")"
        );
    }    
}

void MicrowaveHandler::MicrowaveAwgHandler::open_connection() {
    std::lock_guard<std::mutex> lock(this->awg_mtx);
    this->awg.open_connection();
    
    this->max_step_size = this->awg.get_max_step_count();
    this->max_segment_count = min(this->max_segment_count, this->awg.get_max_segment_count());

    // Set the initial step index to start from
    this->awg.set_initial_step(MW_START_STEP_INDEX);

    // Uploading the short segments used for start/end markers
    awg.init_segment(MW_SHORT_SEGMENT_INDEX, this->min_segment_size);

    auto short_buffer = this->awg.allocate_transfer_buffer(this->min_segment_size);
    awg.fill_transfer_buffer(short_buffer, this->min_segment_size, 0);
    awg.load_data(MW_SHORT_SEGMENT_INDEX, *short_buffer, this->min_segment_size, true);

    awg.seqmem_update(MW_START_STEP_INDEX, MW_SHORT_SEGMENT_INDEX, 1,
                      MW_START_STEP_INDEX, SPCSEQ_ENDLOOPALWAYS);

    awg.seqmem_update(MW_END_STEP_INDEX, MW_SHORT_SEGMENT_INDEX, 1,
                      MW_END_STEP_INDEX, SPCSEQ_ENDLOOPALWAYS);

}

void MicrowaveHandler::MicrowaveAwgHandler::close_connection() {
    std::lock_guard<std::mutex> lock(this->awg_mtx);
    this->awg.close_card();
}

void MicrowaveHandler::MicrowaveAwgHandler::force_trigger() {
    std::lock_guard<std::mutex> lock(this->awg_mtx);
    this->awg.force_hardware_trigger();
}

void MicrowaveHandler::MicrowaveAwgHandler::start() {
    std::lock_guard<std::mutex> lock(this->awg_mtx);
    // Move to the end step
    this->awg.seqmem_update(
        MW_START_STEP_INDEX,
        MW_SHORT_SEGMENT_INDEX,
        1,
        this->step_to_run_index,
        SPCSEQ_ENDLOOPONTRIG
    );
    this->awg.start_stream();

    this->flag_timer_worker_active.store(true);
}

void MicrowaveHandler::MicrowaveAwgHandler::stop() {
    std::lock_guard<std::mutex> lock(this->awg_mtx);
    this->awg.stop_card();
}

bool MicrowaveHandler::MicrowaveAwgHandler::is_connected() {
    std::lock_guard<std::mutex> lock(this->awg_mtx);
    return this->awg.is_connection_open();
}

MicrowaveHandler::MicrowaveAwgHandler::~MicrowaveAwgHandler() {
    this->flag_timer_worker_kill.store(true);
    if (this->timer_worker_thread && this->timer_worker_thread->joinable()) {
        this->timer_worker_thread->join();
    }
    if (awg.is_connection_open()) {
        this->close_connection();
    }
}

void MicrowaveHandler::MicrowaveAwgHandler::reload(bool flag_translate) {
    this->synthesizer.reload(flag_translate);
}

std::tuple<
    std::vector<MicrowaveHandler::MicrowaveAwgHandler::IQMixerWaveform>, 
    std::vector<int>
> MicrowaveHandler::MicrowaveAwgHandler::breakdown_waveforms(const std::vector<Waveform>& waveforms) {
    
    std::vector<MicrowaveHandler::MicrowaveAwgHandler::IQMixerWaveform> waveforms_list;
    std::vector<int> repetitions_list;

    double t = 0;
    double dt = 1 / this->awg.get_sample_rate();
    double t_initial_pause = 0;
    size_t N = waveforms.size();

    for (size_t i = 0; i < N; ++i) {

        if ( auto *p = boost::get<Pause>(&waveforms[i]) ) {

            if (i == N - 1 || boost::get<Pause>(&waveforms[i + 1]) != nullptr) {
                t_initial_pause += p->duration;
                continue;
            }
            
            auto pause_time = p->duration + t_initial_pause;
            int sample_count = round(pause_time / dt);

            if (sample_count < this->default_pause_segment_size) {
                t_initial_pause += p->duration;
                continue;
            }

            int repetitions = sample_count / this->default_pause_segment_size;
            int remaining_samples = sample_count % this->default_pause_segment_size;

            if (repetitions > MW_MAX_STEP_REPETITION) {
                throw std::runtime_error("Pause repetition count exceeds hardware limit");
            }

            auto pause_duration = this->default_pause_segment_size * dt;

            waveforms_list.push_back(
                MicrowaveHandler::MicrowaveAwgHandler::IQMixerWaveform(
                    Pause(
                        pause_duration
                    ),
                    t,
                    0,
                    pause_duration
                )
            );
            t += pause_duration * repetitions;
            repetitions_list.push_back(repetitions);
            t_initial_pause = remaining_samples * dt;
        
        } else if (auto *p = boost::get<SquarePulse>(&waveforms[i])) {

            auto duration = p->duration + t_initial_pause + abs(this->digital_offset_time);
            int sample_count = round(duration / dt);
            auto dsample_count = ( sample_count % this->segment_size_steps == 0 ? 0 : this->segment_size_steps - (sample_count % this->segment_size_steps) );

            sample_count += dsample_count;
            duration += dsample_count * dt;

            if (sample_count < this->min_segment_size) {
                continue;
            }

            waveforms_list.push_back(
                MicrowaveHandler::MicrowaveAwgHandler::IQMixerWaveform(
                    waveforms[i],
                    t,
                    t_initial_pause,
                    duration
                )
            );
            repetitions_list.push_back(1);

            t += duration;
            t_initial_pause = 0;

        } else {

            throw std::runtime_error("Unsupported waveform type in breakdown_waveforms: " + std::to_string(waveforms[i].which()) );

        }
    }
    
    return {waveforms_list, repetitions_list};
}

int MicrowaveHandler::MicrowaveAwgHandler::upload_iqmixer_waveform(IQMixerWaveform iqmixer_waveform, bool lock_awg) {
    std::unique_lock<std::mutex> lock(this->awg_mtx, std::defer_lock);
    if (lock_awg) {
        lock.lock();
    }

    const auto& hash = iqmixer_waveform.hash;

    // Find appropriate segment index
    int segment_index;
    bool is_uploaded;
    if (this->awg_segments_queue.contains_hash(hash)) { // If the waveform has already been uploaded

        this->awg_segments_queue.touch_hash(hash);
        segment_index = this->hash_segment_index_map[hash];
        is_uploaded = true;

    } else { // If the waveform has not been uploaded yet
        
        is_uploaded = false;
        
        if (this->awg_segments_queue.size() == this->max_segment_count) { // Maximum number of segments reached
            auto evicted_hash = this->awg_segments_queue.remove_oldest_hash();
            segment_index = this->hash_segment_index_map[evicted_hash];
            this->hash_segment_index_map.erase(evicted_hash);
            
            this->hash_segment_index_map[hash] = segment_index;
            this->awg_segments_queue.add_hash(hash);

        } else { // Still have free memory for new segments

            segment_index = MW_INITIAL_SEGMENT_INDEX + this->awg_segments_queue.size();
            this->awg_segments_queue.add_hash(hash);
            this->hash_segment_index_map[hash] = segment_index;

        }
    }

    // If not uploaded yet, synthesize the waveforms and upload
    if (!is_uploaded) {
        int sample_count = round(iqmixer_waveform.duration * this->awg.get_sample_rate());
        auto buffer = this->awg.allocate_transfer_buffer(sample_count);
        this->awg.init_segment(segment_index, sample_count);
        
        if (auto *p = boost::get<Pause>(&iqmixer_waveform.waveform)) {
            this->awg.fill_transfer_buffer(buffer, sample_count, 0);
        } else {
            this->synthesizer.generate_pulse(
                *buffer,
                sample_count,
                this->awg,
                iqmixer_waveform.waveform,
                iqmixer_waveform.time,
                iqmixer_waveform.t_initial_pause
            );
        }

        this->awg.load_data(segment_index, *buffer, sample_count, false);
    }

    return segment_index;
}

void MicrowaveHandler::MicrowaveAwgHandler::clear_memory() {
    this->next_step_to_load_index = MW_INITIAL_STEP_INDEX;
    this->step_to_run_index       = MW_END_STEP_INDEX;

    this->hash_segment_index_map.clear();
    this->awg_segments_queue.clear();
}

int MicrowaveHandler::MicrowaveAwgHandler::increment_step_index(int index, int step_size) {
    return (index - MW_INITIAL_STEP_INDEX + step_size) % (this->max_step_size - MW_INITIAL_STEP_INDEX) + MW_INITIAL_STEP_INDEX;
}

void MicrowaveHandler::MicrowaveAwgHandler::upload_waveforms(
    const std::vector<MicrowaveHandler::Waveform>& waveforms)
{
    std::lock_guard<std::mutex> lock(this->awg_mtx);

    INFO << "[AWG] upload_waveforms() ENTER, waveforms.size() = "
         << waveforms.size() << std::endl;

    // Breakdown
    INFO << "[AWG] Breaking down waveforms..." << std::endl;
    auto iqmixer_waveforms_tuple = this->breakdown_waveforms(waveforms);

    const auto& iqmixer_waveforms = std::get<0>(iqmixer_waveforms_tuple);
    const auto& repetitions      = std::get<1>(iqmixer_waveforms_tuple);
    const size_t N = iqmixer_waveforms.size(); // Number of segments to upload

    INFO << "[AWG] Breakdown complete. N = " << N << std::endl;

    if (N == 0) {
        INFO << "[AWG] No waveforms. Resetting indices and returning." << std::endl;
        this->step_to_run_index       = MW_END_STEP_INDEX;
        return;
    }

    // Upload segments
    INFO << "[AWG] Uploading " << N << " IQ mixer waveforms..." << std::endl;

    std::vector<int> segment_indices(N);
    for (size_t i = 0; i < N; ++i) {
        INFO << "[AWG] Uploading segment " << i << " / " << (N - 1) << std::endl;
        segment_indices[i] = this->upload_iqmixer_waveform(iqmixer_waveforms[i]);
        INFO << "[AWG] Segment " << i << " uploaded, index = "
             << segment_indices[i] << std::endl;
        INFO << boost::apply_visitor([](auto&& a) {return a.to_string();}, iqmixer_waveforms[i].waveform) << std::endl;
    }

    INFO << "[AWG] Waiting for data load..." << std::endl;
    this->awg.wait_for_data_load();
    INFO << "[AWG] Data load complete." << std::endl;

    // Sequence memory programming
    const auto last_step_index =
        this->increment_step_index(this->next_step_to_load_index, N - 1);

    INFO << "[AWG] Programming sequence memory. "
         << "First step = " << this->next_step_to_load_index
         << ", Last step = " << last_step_index << std::endl;

    // Last step (terminates sequence)
    INFO << "[AWG] Programming last step (terminating)." << std::endl;
    this->awg.seqmem_update(
        last_step_index,
        segment_indices[N - 1],
        repetitions[N - 1],
        MW_END_STEP_INDEX,
        SPCSEQ_ENDLOOPALWAYS
    );

    // Remaining steps
    for (int i = static_cast<int>(N) - 2; i >= 0; --i) {
        const auto step_index =
            this->increment_step_index(this->next_step_to_load_index, i);
        const auto next_index =
            this->increment_step_index(step_index, 1);

        INFO << "[AWG] Programming step " << step_index
             << " -> next " << next_index
             << ", segment = " << segment_indices[i]
             << ", reps = " << repetitions[i] << std::endl;

        this->awg.seqmem_update(
            step_index,
            segment_indices[i],
            repetitions[i],
            next_index,
            SPCSEQ_ENDLOOPALWAYS
        );
    }

    // Update state
    this->step_to_run_index = this->next_step_to_load_index;
    this->next_step_to_load_index =
        this->increment_step_index(last_step_index, 1);

    INFO << "[AWG] upload_waveforms() EXIT. "
         << "step_to_run_index = " << this->step_to_run_index
         << ", next_step_to_load_index = "
         << this->next_step_to_load_index << std::endl;
}

void MicrowaveHandler::MicrowaveAwgHandler::setup_timer_worker() {
    this->streaming_time = 0;
    this->flag_timer_worker_kill.store(false);
    this->flag_timer_worker_active.store(false);
    this->timer_worker_thread = std::make_unique<std::thread>(&MicrowaveAwgHandler::timer_worker, this);
}

void MicrowaveHandler::MicrowaveAwgHandler::timer_worker() {
    while (!this->flag_timer_worker_kill.load()) {
        try {
            if (this->flag_timer_worker_active.load()) {
                auto current_step = this->get_awg_step();
                if (current_step != MW_START_STEP_INDEX) {
                    auto time = std::chrono::steady_clock::now().time_since_epoch();
                    this->streaming_time.store(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(time).count()
                    );
                    this->flag_timer_worker_active.store(false);
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(this->timer_worker_wait_time_ms));
            }
        } catch (const std::exception& e) {
            ERROR << "Exception in timer_worker: " << e.what() << std::endl;
        }
    }   
}

int64_t MicrowaveHandler::MicrowaveAwgHandler::get_streaming_time() {
    return this->streaming_time.load();
}