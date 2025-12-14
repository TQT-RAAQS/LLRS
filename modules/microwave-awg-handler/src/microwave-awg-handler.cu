#include "microwave-awg-handler.h"

using namespace MicrowaveWaveforms;

MicrowaveHandler::MicrowaveAwgHandler::MicrowaveAwgHandler(const std::string& handler_config, const std::string& awg_config) {
    this->reload();
    this->awg = AWG(awg_config);

    auto config = YAML::LoadFile(MICROWAVE_AWG_HANDLER_CONFIG(handler_config));
    this->max_segment_count = config["max_segment_count"].as<int>();
    this->default_pause_segment_size = config["default_pause_segment_size"].as<int>();
    this->digital_offset_time = config["digital_offset_time"].as<double>();
    this->min_segment_size = this->awg.get_minimum_segment_size();
    this->segment_size_steps = this->awg.get_segment_size_steps();

    this->synthesizer.set_digital_offset_time(this->digital_offset_time);
}

void MicrowaveHandler::MicrowaveAwgHandler::open_connection() {
    this->awg.open_connection();
    
    this->max_step_size = this->awg.get_max_step_count();
    this->max_segment_count = min(this->max_segment_count, this->awg.get_max_segment_count());

    // Set the initial step index to start rom
    this->awg.set_initial_step(MW_START_STEP_INDEX);

    // Uploading the short segments used for start/end markers
    awg.init_segment(MW_START_SEGMENT_INDEX, this->min_segment_size);
    awg.init_segment(MW_END_SEGMENT_INDEX, this->min_segment_size);

    auto short_buffer = this->awg.allocate_transfer_buffer(this->min_segment_size);
    awg.fill_transfer_buffer(short_buffer, this->min_segment_size, 0);
    awg.load_data(MW_START_SEGMENT_INDEX, *short_buffer, this->min_segment_size, true);
    awg.load_data(MW_END_SEGMENT_INDEX, *short_buffer, this->min_segment_size, true);

    awg.seqmem_update(MW_START_STEP_INDEX, MW_START_SEGMENT_INDEX, 1,
                      MW_START_STEP_INDEX, SPCSEQ_ENDLOOPALWAYS);

    awg.seqmem_update(MW_END_STEP_INDEX, MW_END_SEGMENT_INDEX, 1,
                      MW_END_STEP_INDEX, SPCSEQ_ENDLOOPALWAYS);

}

void MicrowaveHandler::MicrowaveAwgHandler::close_connection() {
    this->awg.close_card();
}

void MicrowaveHandler::MicrowaveAwgHandler::start() {
    this->awg.seqmem_update(
        MW_START_STEP_INDEX,
        MW_START_SEGMENT_INDEX,
        1,
        this->step_to_run_index,
        SPCSEQ_ENDLOOPALWAYS
    );
    this->awg.start_stream();
}

void MicrowaveHandler::MicrowaveAwgHandler::stop() {
    this->awg.stop_card();
}

bool MicrowaveHandler::MicrowaveAwgHandler::is_connected() const {
    return this->awg.is_connection_open();
}

MicrowaveHandler::MicrowaveAwgHandler::~MicrowaveAwgHandler() {
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

int MicrowaveHandler::MicrowaveAwgHandler::upload_iqmixer_waveform(IQMixerWaveform iqmixer_waveform) {
    const auto& hash = iqmixer_waveform.hash;

    // Find appropriate segment index
    int segment_index;
    bool is_uploaded;
    if (this->awg_segments_queue.contains_hash(hash)) { // If th3e waveform has already been uploaded

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

    // Synthesize the waveforms and upload
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
    const std::vector<MicrowaveWaveforms::Waveform>& waveforms)
{
    // Breakdown
    auto iqmixer_waveforms_tuple = this->breakdown_waveforms(waveforms);

    const auto& iqmixer_waveforms = std::get<0>(iqmixer_waveforms_tuple);
    const auto& repetitions      = std::get<1>(iqmixer_waveforms_tuple);
    const size_t N = iqmixer_waveforms.size();

    if (N == 0) {
        this->next_step_to_load_index = MW_INITIAL_STEP_INDEX;
        this->step_to_run_index       = MW_END_STEP_INDEX;
        return;
    }

    // Upload segments
    std::vector<int> segment_indices(N);
    for (size_t i = 0; i < N; ++i) {
        segment_indices[i] = this->upload_iqmixer_waveform(iqmixer_waveforms[i]);
    }
    this->awg.wait_for_data_load();

    // Sequence memory programming
    const auto last_step_index =
        this->increment_step_index(this->next_step_to_load_index, N - 1);

    // Last step (terminates sequence)
    this->awg.seqmem_update(
    last_step_index,
    segment_indices[N - 1],
    repetitions[N - 1],
    MW_END_STEP_INDEX,
    SPCSEQ_ENDLOOPALWAYS
    );

    // Remaining steps
    for (int i = static_cast<int>(N) - 2; i >= 0; --i) {
    const auto step_index = this->increment_step_index(this->next_step_to_load_index, i);
    const auto next_index = this->increment_step_index(step_index, 1);

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
}