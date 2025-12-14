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
}

void MicrowaveHandler::MicrowaveAwgHandler::open_connection() {
    this->awg.open_connection();
    
    this->max_step_size = this->awg.get_max_step_count();
    this->max_segment_count = min(this->max_segment_count, this->awg.get_max_segment_count());

    // Uploading the short segments used for start/end markers
    awg.init_segment(MW_START_SEGMENT_INDEX, this->min_segment_size);
    awg.init_segment(MW_END_SEGMENT_INDEX, this->min_segment_size);

    auto short_buffer = this->awg.allocate_transfer_buffer(this->min_segment_size);
    awg.fill_transfer_buffer(short_buffer, this->min_segment_size, 0);
    awg.load_data(MW_START_SEGMENT_INDEX, *short_buffer, this->min_segment_size, true);
    awg.load_data(MW_END_SEGMENT_INDEX, *short_buffer, this->min_segment_size, true);

    // Uploading the pause segment
    MicrowaveHandler::MicrowaveAwgHandler::IQMixerWaveform pause_waveform(
        Pause(
            this->default_pause_segment_size / this->awg.get_sample_rate()
        ),
        0,
        0
    );
    this->upload_iqmixer_waveform(pause_waveform);
}

void MicrowaveHandler::MicrowaveAwgHandler::close_connection() {
    this->awg.close_card();
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
    if (flag_translate) {
        this->config_translator.translate_iqmixer();
    }

    std::ifstream infile(IQMIXER_TRANSLATION_FILE, std::ios::binary);
    
    infile.read(reinterpret_cast<char*>(&this->dphi), sizeof(double));
    infile.read(reinterpret_cast<char*>(&this->vI_dc), sizeof(double));
    infile.read(reinterpret_cast<char*>(&this->vQ_dc), sizeof(double));

    infile.close();
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

            waveforms_list.push_back(
                MicrowaveHandler::MicrowaveAwgHandler::IQMixerWaveform(
                    Pause(
                        this->default_pause_segment_size * dt
                    ),
                    t,
                    0
                )
            );
            repetitions_list.push_back(repetitions);
            t_initial_pause = remaining_samples * dt;
        
        } else if (auto *p = boost::get<SquarePulse>(&waveforms[i])) {

            auto sample_count = round((p->duration + t_initial_pause) / dt);
            if (sample_count < this->min_segment_size) {
                INFO << "Eliminating a square pulse of duration " << p->duration*1e9 << " ns due to minimum segment size constraint.\n";
                t += p->duration + t_initial_pause;
                continue;
            }

            waveforms_list.push_back(
                MicrowaveHandler::MicrowaveAwgHandler::IQMixerWaveform(
                    waveforms[i],
                    t,
                    t_initial_pause
                )
            );
            repetitions_list.push_back(1);

            t += p->duration + abs(this->digital_offset_time) + t_initial_pause;
            t_initial_pause = 0;

        } else {

            throw std::runtime_error("Unsupported waveform type in breakdown_waveforms: " + std::to_string(waveforms[i].which()) );

        }
    }
    
    return {waveforms_list, repetitions_list};
}

void MicrowaveHandler::MicrowaveAwgHandler::upload_iqmixer_waveform(IQMixerWaveform iqmixer_waveform) {
    const auto& hash = iqmixer_waveform.hash;

    // Find appropriate segment index
    int segment_index;
    bool is_uploaded;
    if (this->awg_segments_queue.containsHash(hash)) { // If the waveform has already been uploaded

        this->awg_segments_queue.touchHash(hash);
        segment_index = this->hash_segment_index_map[hash];
        is_uploaded = true;

    } else { // If the waveform has not been uploaded yet
        
        is_uploaded = false;
        
        if (this->awg_segments_queue.size() == this->max_segment_count) { // Maximum number of segments reached
            
            auto evicted_hash = this->awg_segments_queue.removeOldestHash();
            segment_index = this->hash_segment_index_map[evicted_hash];
            this->hash_segment_index_map.erase(evicted_hash);
            
            this->hash_segment_index_map[hash] = segment_index;
            this->awg_segments_queue.addHash(hash);

        } else { // Still have free memory for new segments

            segment_index = MW_INITIAL_STEP_INDEX + this->awg_segments_queue.size();
            this->awg_segments_queue.addHash(hash);
            this->hash_segment_index_map[hash] = segment_index;

        }
    }

    // Synthesize the waveforms and upload
    if (is_uploaded) return;

    
}

void MicrowaveHandler::MicrowaveAwgHandler::upload_waveforms(const std::vector<MicrowaveWaveforms::Waveform> &waveforms) {

    auto iqmixer_waveforms = this->breakdown_waveforms(waveforms);
}