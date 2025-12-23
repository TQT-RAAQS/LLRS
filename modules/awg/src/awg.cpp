/**
 * @brief Handles configurations for the AWG class
 * @date Dec 2023
 */

#include "awg.hpp"
#include <bits/stdc++.h>

/**
 * @brief Constructor for AWG class
 */
AWG::AWG(const std::string& config_name) { 
    if (read_config(AWG_CONFIG_PATH(config_name)) != AWG_OK) {
        std::cerr << "Error occured in parsing AWG config.\n";
        throw std::runtime_error("Could not parse the AWG config.\n");
    }
}

/**
 * @brief Destructor for AWG class
 */
AWG::~AWG() {
    if (flag_is_connected) {
        stop_card();
        close_card();
    }
}

/**
 * @brief Parse AWG configurations from YAML file
 * @param filename name of file
 * @return status code
 */
int AWG::read_config(std::string filename) {
    YAML::Node node;

    /// Open file
    try {
        node = YAML::LoadFile(filename);
    } catch (const YAML::BadFile &e) {
        std::cerr << "Error loading YAML file (awg.cpp)." << std::endl;
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    /// Extract file contents
    config.driver_path = node["driver_path"].as<std::string>();

    config.external_clock_flag = node["external_clock_flag"].as<bool>();
    if (config.external_clock_flag) {
        config.external_clock_freq = node["external_clock_freq"].as<int>();
    }
    else {
        config.external_clock_freq = 0;
    }
    config.channels = node["channels"].as<std::vector<int>>();
    for (const auto c : config.channels) {
        config.channel_bit_shifts[c] = 0;
        config.channel_digout_indices[c] = std::vector<int>();
    }
    this->num_channels = config.channels.size();
    config.amp = node["amp"].as<std::vector<int>>();
    config.awg_num_segments = node["awg_num_segments"].as<int>();
    config.sample_rate = node["sample_rate"].as<double>();
    std::string wfm_mask_str = node["wfm_mask"].as<std::string>();
    config.wfm_mask = std::stoi(wfm_mask_str, 0, 16);
    config.trigger_size = node["trigger_size"].as<int>();
    config.vpp = node["vpp"].as<int>();
    config.acq_timeout = node["acq_timeout"].as<int>();
    config.idle_segment_wfm = node["idle_segment_wfm"].as<bool>();
    config.waveforms_per_segment = node["waveforms_per_segment"].as<int>();
    config.null_seg_num_waveforms = node["null_seg_num_waveforms"].as<int>();
    config.idle_seg_num_waveforms = node["idle_seg_num_waveforms"].as<int>();
    config.samples_per_segment = 0;

    const auto& itcn = node["input_trigger"];
    input_trigger_config_t itc;
    itc.ports =       itcn["ports"].as<std::vector<size_t>>();
    itc.logic =       itcn["logic"].as<uint8_t>();
    itc.edge  =       itcn["edge"].as<uint8_t>();
    itc.rearm =       static_cast<uint8_t>(itcn["rearm"].as<bool>());
    itc.level_0_mv  = itcn["level_0_mv"].as<std::vector<int32>>();
    itc.level_1_mv  = itcn["level_1_mv"].as<std::vector<int32>>();
    itc.timeout_ms  = itcn["timeout_ms"].as<int32>();
    config.input_trigger_config = std::move(itc);

    config.first_step_index = node["first_step_index"].as<int32>();

    config.async_out_trig_channels = node["async_out_trig_channels"].as<std::vector<int8_t>>();

    const auto& sdoc = node["sync_out_trig"];
    if (!sdoc.IsSequence()) {
        throw std::runtime_error("The type of the sync_out_trig config must be a list.");
    }
    for (size_t i = 0; i < sdoc.size(); ++i) {
        sync_output_trigger_config_t c;
        c.port    = sdoc[i]["port"].as<int8_t>();
        c.channel = sdoc[i]["channel"].as<int8_t>();
        c.bit     = sdoc[i]["bit"].as<int8_t>();
        
        if (config.channel_bit_shifts.find(c.channel) == config.channel_bit_shifts.end()) {
            throw std::runtime_error("The channel configured for sync output trigger is not enabled as an analog output channel: " + std::to_string(c.channel));
        }

        config.sync_out_trig_configs.push_back(c);

        config.channel_bit_shifts[c.channel] = std::max(16 - c.bit, config.channel_bit_shifts[c.channel]);
        config.channel_digout_indices[c.channel].push_back(i);
    }

    return 0;
}

/**
 * @brief Configure the AWG
 * @return Error code
 */
int AWG::open_connection() {
    p_card = spcm_hOpen(config.driver_path.c_str());
    if (!p_card) {
        std::cerr << "ERROR: AWG Card not found.\n";
        return 1;
    } else {
        reset_card();
    }
    int status = 0;
    status |= enable_channels(config.channels);
    status |= enable_outputs(config.channels, config.amp);
    status |= set_sample_rate(config.sample_rate);
    if (config.external_clock_flag) {
        status |= set_external_clock_mode(config.external_clock_freq);
    } else {
        status |= set_internal_clock_mode();
    }
    status |= set_input_trigger_settings(config.input_trigger_config);

    status |= setup_async_output_triggers(config.async_out_trig_channels);
    status |= setup_sync_output_triggers(config.sync_out_trig_configs);

    status |= spcm_dwGetParam_i32(p_card, SPC_SEQMODE_AVAILMAXSTEPS, &max_step);
    status |= spcm_dwGetParam_i32(p_card, SPC_SEQMODE_AVAILMAXSEGMENT, &max_segment);
    status |= spcm_dwGetParam_i32(p_card, SPC_MIINST_BYTESPERSAMPLE, &bps);
    status |= spcm_dwGetParam_i32(p_card, SPC_CHCOUNT, &lSetChannels);
    dwFactor = 1;

    if (config.awg_num_segments == 0 ||
        ((config.awg_num_segments & (config.awg_num_segments - 1)) != 0) ||
        config.awg_num_segments > max_step) {
        std::cerr << "ERROR: AWG Constructor -> max_seg needs to be "
                     "a power of 2 and less than "
                  << max_step << std::endl;
        return 1;
    }
    if (config.first_step_index >= config.awg_num_segments) {
        std::cerr << "ERROR: The index of the starting step should be smaller than the maximum number of programmed segments.\n";
        std::cerr << config.first_step_index << " must be smaller than " <<  config.awg_num_segments << std::endl;
    }

    status |= spcm_dwSetParam_i32(p_card, SPC_CARDMODE, SPC_REP_STD_SEQUENCE);
    status |= spcm_dwSetParam_i32(p_card, SPC_SEQMODE_MAXSEGMENTS,
                                  config.awg_num_segments);
    status |= this->set_initial_step(config.first_step_index);

    assert(config.samples_per_segment <=
           (AWG_MEMORY_SIZE / bps) /
               config.awg_num_segments); // assert that the samples per segments
                                         // matches the size of the AWG memory

    /// Allocate the continuous buffer
    spcm_dwGetContBuf_i64(p_card, SPCM_BUF_DATA, &continuousBuffer,
                          &continuousBufferSize);
    std::cerr << "Physically continuous buffer of size " << continuousBufferSize
              << " was successfully allocated." << std::endl;

    flag_is_connected = true;
    return status;
}

int AWG::set_initial_step(int32 step) {
    return spcm_dwSetParam_i32(p_card, SPC_SEQMODE_STARTSTEP, step);
}

// This function forces a "hardware trigger" event.
// Example: If the AWG is at step 1, which points to step 2 on trigger, then
// this function would make this jump even if an actual hardware trigger is not
// sent.
void AWG::force_hardware_trigger() {
    spcm_dwSetParam_i32(p_card, SPC_M2CMD, M2CMD_CARD_FORCETRIGGER);
}

void AWG::configure_segment_length(double waveform_duration) {
    config.waveform_length =
        static_cast<int>(config.sample_rate * waveform_duration);
    config.null_segment_length =
        config.waveform_length * config.null_seg_num_waveforms;
    config.idle_segment_length =
        config.waveform_length * config.idle_seg_num_waveforms;
    config.samples_per_segment =
        config.waveforms_per_segment * config.waveform_length;
}

/**
 * @brief Enables specified channels
 * @param channels Vector of channels to be enabled
 * @return Error code
 */
int AWG::enable_channels(const std::vector<int> &channels) {

    int tag = 0;
    /// update tag with the index of each channel to be turned on
    for (size_t i = 0; i < channels.size(); ++i) {
        switch (channels[i]) {
        case 0:
            tag |= CHANNEL0;
            break;
        case 1:
            tag |= CHANNEL1;
            break;
        case 2:
            tag |= CHANNEL2;
            break;
        case 3:
            tag |= CHANNEL3;
            break;
        default:
            std::cerr << "ERROR: AWG channel not supported\n";
            break;
        }
    }

    num_channels = channels.size();
    return spcm_dwSetParam_i32(p_card, SPC_CHENABLE, tag);
}

/**
 * @brief Enables specified output channels
 * @param channels Vector of output channels to be enabled
 * @param amp Vector of amplitudes corresponding to each output channel
 * @return Error code
 */
int AWG::enable_outputs(const std::vector<int> &channels,
                        const std::vector<int> &amp) {

    assert(channels.size() == amp.size());
    int status = 0;
    for (size_t i = 0; i < channels.size(); ++i) {
        switch (channels[i]) {
        case 0:
            status |= spcm_dwSetParam_i32(p_card, SPC_AMP0, amp[i]);
            status |= spcm_dwSetParam_i32(p_card, SPC_ENABLEOUT0, 1);
            break;
        case 1:
            status |= spcm_dwSetParam_i32(p_card, SPC_AMP1, amp[i]);
            status |= spcm_dwSetParam_i32(p_card, SPC_ENABLEOUT1, 1);
            break;
        case 2:
            status |= spcm_dwSetParam_i32(p_card, SPC_AMP2, amp[i]);
            status |= spcm_dwSetParam_i32(p_card, SPC_ENABLEOUT2, 1);
            break;
        case 3:
            status |= spcm_dwSetParam_i32(p_card, SPC_AMP3, amp[i]);
            status |= spcm_dwSetParam_i32(p_card, SPC_ENABLEOUT3, 1);
            break;
        default:
            std::cerr << "ERROR: AWG channel not supported\n";
            return 1;
        }
    }

    return status;
}

/**
 * @brief Set the sample rate
 * @param sample_rate sample rate
 * @return Error code
 */
int AWG::set_sample_rate(int sample_rate) {
    return spcm_dwSetParam_i64(p_card, SPC_SAMPLERATE, sample_rate);
}

/**
 * @brief Sets the external clock mode
 * @param external_clock_freq Desired frequency of the external clock
 * @return Error code
 */
int AWG::set_external_clock_mode(int external_clock_freq) {
    int status = 0;
    status |= spcm_dwSetParam_i32(p_card, SPC_CLOCKMODE, SPC_CM_EXTREFCLOCK);
    status |=
        spcm_dwSetParam_i32(p_card, SPC_REFERENCECLOCK, external_clock_freq);
    return status;
}

/**
 * @brief Sets the internal clock mode
 * @param
 * @return Error code
 */
int AWG::set_internal_clock_mode() {
    return spcm_dwSetParam_i32(p_card, SPC_CLOCKMODE, SPC_CM_INTPLL);
}

/**
 * @brief Sets upper and lower trigger levels and the mode of the trigger
 * @return Error code
 */
int AWG::set_input_trigger_settings(const input_trigger_config_t& config) {
    int status = 0;

    // Setting level 0 and 1 voltages
    for (size_t i = 0; i < config.ports.size(); ++i) {
        int ch = config.ports[i];
    
        int32_t lvl0 = config.level_0_mv.at(i);
        int32_t lvl1 = config.level_1_mv.at(i);
    
        if (ch == 0) {
            status |= spcm_dwSetParam_i32(p_card, SPC_TRIG_EXT0_LEVEL0, lvl0);
            status |= spcm_dwSetParam_i32(p_card, SPC_TRIG_EXT0_LEVEL1, lvl1);
        } else if (ch == 1) {
            status |= spcm_dwSetParam_i32(p_card, SPC_TRIG_EXT1_LEVEL0, lvl0);
            status |= spcm_dwSetParam_i32(p_card, SPC_TRIG_EXT1_LEVEL1, lvl1);
        } else {
            throw std::runtime_error("Channel not supported: " + std::to_string(ch));
        }
    }    

    // Trigger edge status
    auto edge_status = config.edge ? SPC_TM_POS : SPC_TM_NEG ;
    for (const auto ch : config.ports) {
        if (ch == 0) {
            auto edge_status_ch0 = edge_status | (config.rearm ? SPC_TM_REARM : 0b0);

            status |= spcm_dwSetParam_i32(p_card, SPC_TRIG_EXT0_MODE, edge_status_ch0);
        } else if (ch == 1) {
            status |= spcm_dwSetParam_i32(p_card, SPC_TRIG_EXT1_MODE, edge_status);
        } else {
            throw std::runtime_error("Channel not supported: " + std::to_string(ch));
        }
    }

    // Logic mask
    auto trig_mask = config.logic ? SPC_TRIG_ANDMASK : SPC_TRIG_ORMASK;
    int32 logic_ch_mask = 0;
    for (const auto ch : config.ports) {
        if (ch == 0) {
            logic_ch_mask |= SPC_TMASK_EXT0;
        } else if (ch == 1) {
            logic_ch_mask |= SPC_TMASK_EXT1;
        } else {
            throw std::runtime_error("Channel not supported: " + std::to_string(ch));
        }
    }
    status |= spcm_dwSetParam_i32(p_card, trig_mask, logic_ch_mask);

    // Timeout
    status |= spcm_dwSetParam_i32(p_card, SPC_TIMEOUT, config.timeout_ms);

    return status;
}

/**
 * @brief Set MPIO line as SYNC digital output with the analog channels. Digital
 * logical 1 is set for each sample as the most significant bit.
 * @return Error code
 */
int AWG::set_dout_trigger_mode(int32 line, int32 channel) {
    int trigger_mode =
        (SPCM_XMODE_DIGOUT | channel | SPCM_XMODE_DIGOUTSRC_BIT15);
    return spcm_dwSetParam_i32(p_card, line, trigger_mode);
}

/**
 * @brief Set MPIO line as ASYNCOUT for external triggers. M2CMD_CARD_START
 * needs to be called after a change in setting.
 * @return Error code
 */
int AWG::setup_async_output_triggers(std::vector<int8_t> channels) {
    int status = 0;
    for (const auto c : channels) {
        switch (c) {
            case 0:
                status |= spcm_dwSetParam_i32(p_card, SPCM_X0_MODE, SPCM_XMODE_ASYNCOUT);
                break;
            case 1:
                status |= spcm_dwSetParam_i32(p_card, SPCM_X1_MODE, SPCM_XMODE_ASYNCOUT);
                break;
            case 2:
                status |= spcm_dwSetParam_i32(p_card, SPCM_X2_MODE, SPCM_XMODE_ASYNCOUT);
                break;
            default:
                throw std::runtime_error("This channel is not a valid output channel available for asynchronous triggering: " + std::to_string(c));
        }
    }
    return status;
}

/**
 * @brief Configures synchronous output triggers for the AWG.
 *
 * This function sets up synchronous output triggers on specific ports, channels, 
 * and bits as defined in the provided configuration. Synchronous triggers are 
 * used to output digital signals that are synchronized with the analog waveform 
 * generation, enabling precise control over external devices or other AWGs.
 *
 * @param configs A vector of `sync_output_trigger_config_t` structures, where each 
 *        structure specifies the configuration for a single synchronous output trigger.
 *        Each configuration includes:
 *        - `port`: The output port (0, 1, or 2).
 *        - `channel`: The channel to be used for the trigger (0, 1, 2, or 3).
 *        - `bit`: The bit to be used for the trigger (13, 14, or 15).
 *
 * @return int Status code indicating the success or failure of the operation.
 *         - Returns 0 on success.
 *         - Returns a non-zero value if any hardware configuration call fails.
 *
 * @throws std::runtime_error If an invalid port, channel, or bit is specified in the configuration.
 *
 * @details
 * - The function iterates through the provided configurations and applies each one 
 *   to the corresponding hardware port.
 * - The `trigger_mode` is constructed by combining the base mode (`SPCM_XMODE_DIGOUT`) 
 *   with the specified channel and bit.
 * - The function validates the port, channel, and bit values, throwing an exception 
 *   if any are invalid.
 * - The hardware configuration is applied using `spcm_dwSetParam_i32`.
 *
 * @example
 * // Example usage:
 * std::vector<sync_output_trigger_config_t> configs = {
 *     {0, 1, 13},  // Configure port 0, channel 1, bit 13
 *     {1, 2, 14},  // Configure port 1, channel 2, bit 14
 * };
 * int status = awg.setup_sync_output_triggers(configs);
 * if (status != 0) {
 *     std::cerr << "Failed to configure synchronous output triggers." << std::endl;
 * }
 */
int AWG::setup_sync_output_triggers(std::vector<sync_output_trigger_config_t> configs) {
    int status = 0;

    int32 port;
    int32 trigger_mode;
    for (const auto c : configs) {
        switch (c.port) {
            case 0:
                port = SPCM_X0_MODE;
                break;
            case 1:
                port = SPCM_X1_MODE;
                break;
            case 2:
                port = SPCM_X2_MODE;
                break;
            default:
                throw std::runtime_error("Undefined output port: " + std::to_string(c.port));
        }

        trigger_mode = SPCM_XMODE_DIGOUT;

        switch (c.channel) {
            case 0:
                trigger_mode |= SPCM_XMODE_DIGOUTSRC_CH0;
                break;
            case 1:
                trigger_mode |= SPCM_XMODE_DIGOUTSRC_CH1;
                break;
            case 2:
                trigger_mode |= SPCM_XMODE_DIGOUTSRC_CH2;
                break;
            case 3:
                trigger_mode |= SPCM_XMODE_DIGOUTSRC_CH3;
                break;
            default:
                throw std::runtime_error("Invalid channel number: " + std::to_string(c.channel));
        }

        switch (c.bit) {
            case 13:
                trigger_mode |= SPCM_XMODE_DIGOUTSRC_BIT13;
                break;
            case 14:
                trigger_mode |= SPCM_XMODE_DIGOUTSRC_BIT14;
                break;
            case 15:
                trigger_mode |= SPCM_XMODE_DIGOUTSRC_BIT15;
                break;
            default:
                throw std::runtime_error("Invalid bit: " + std::to_string(c.bit));
        }
        
        status |= spcm_dwSetParam_i32(p_card, port, trigger_mode);
    }

    return status;
}

/**
 * @brief Async pulse is generated on all MPIOs that are set as
 * SPMCM_XMODE_ASYNCOUT
 * @return Error code
 */
void AWG::generate_async_output_pulse(TriggerType port) {
    spcm_dwSetParam_i32(p_card, SPCM_XX_ASYNCIO, 0); // Set trigger state to 0
    spcm_dwSetParam_i32(p_card, SPCM_XX_ASYNCIO, port); // Set trigger for the designated port to 1.
    spcm_dwSetParam_i32(p_card, SPCM_XX_ASYNCIO, 0); // Reset the trigger state to 0 immediately; this produces a very short trigger (SK measured a ~10 us long trigger on 2025-12-12 on a 4 channel AWG on the port X0).
}

/**
 * @brief Software trigger on card start
 * @return Error code
 */
int AWG::start_stream() {
    return spcm_dwSetParam_i32(p_card, SPC_M2CMD,
                               M2CMD_CARD_START | M2CMD_CARD_FORCETRIGGER | M2CMD_CARD_ENABLETRIGGER);
}

/**
 * @brief Reset the card
 * @return Error code
 */
int AWG::reset_card() {
    return spcm_dwSetParam_i32(p_card, SPC_M2CMD, M2CMD_CARD_RESET);
}

/**
 * @brief Stop the card
 * @return Error code
 */
int AWG::stop_card() {
    return spcm_dwSetParam_i32(p_card, SPC_M2CMD, M2CMD_CARD_STOP);
}

/**
 * @brief Close the card
 */
void AWG::close_card() {
    if (flag_is_connected) {
        this->stop_card();
    } 
    spcm_vClose(p_card);
    flag_is_connected = false; // Set flag to false when connection is closed
}

/**
 * @brief: Initialize a step in AWG's sequence memory by combining all the
 * parameters to one int64 bit value
 * @param lStep => Current step
 * @param llSegment => Associated data memory segment
 * @param llLoop => Number of repeated time before condition is checked
 * @param llNext => Next step
 * @param llCondition => end condition (SPCSEQ_ENDLOOPALWAYS,
 * SPCSEQ_ENDLOOPONTRIG, SPCSEQ_END)
 * @return error code
 */
int AWG::seqmem_update(int64 lStep, int64 llSegment, int64 llLoop, int64 llNext,
                       uint64 llCondition) {
    uint64 llValue =
        (llCondition << 32) | (llLoop << 32) | (llNext << 16) | (llSegment);

    return spcm_dwSetParam_i64(p_card, SPC_SEQMODE_STEPMEM0 + lStep, llValue);
}

void AWG::interleave_data(short* target, const std::vector<std::vector<short>> &waveforms, const std::vector<std::vector<int8>>& digital_trigger) {
    // Verification of the input
    if (waveforms.size() != this->num_channels) {
        throw std::runtime_error("Number of provided waveforms does not match the number of enabled channels.");
    }
    if (digital_trigger.size() != this->config.sync_out_trig_configs.size()) {
        throw std::runtime_error("Number of provided digital trigger waveforms does not match the number of configured synchronous output triggers.");
    }

    for (size_t i = 1; i < waveforms.size(); ++i) {
        if (waveforms[i].size() != waveforms[0].size()) {
            throw std::runtime_error("All waveforms must have the same length.");
        }
    }
    for (size_t i = 0; i < digital_trigger.size(); ++i) {
        if (digital_trigger[i].size() != waveforms[0].size()) {
            throw std::runtime_error("All digital trigger waveforms must have the same length as the analog waveforms.");
        }
    }

    // Interleaving
    size_t num_samples = waveforms[0].size();
    for (size_t j = 0; j < this->num_channels; ++j) {
        const auto bit_shift = this->config.channel_bit_shifts[config.channels[j]];
        if (bit_shift == 0) {
            #pragma omp simd
            for (size_t i = 0; i < num_samples; ++i) {
                target[i * this->num_channels + j] = waveforms[j][i];
            }
            continue;
        }
        #pragma omp simd
        for (size_t i = 0; i < num_samples; ++i) {
            short data = static_cast<short>(static_cast<uint16>(waveforms[j][i]) >> bit_shift);
        
            for (const auto ind : config.channel_digout_indices[config.channels[j]]) {
                data |= static_cast<uint16>(digital_trigger[ind][i]) << config.sync_out_trig_configs[ind].bit;
            }
            target[i * this->num_channels + j] = data;
        }
    }
}

/**
 * @brief: Write to segment and returns status
 * @param seg_num => sequence segment number to refer to
 * @param p_data => pointer toa source data
 * @param size => size in number o fsamples to write for each individual channel. DO NOT MULTIPLY BY THE NUMBER OF CHANNELS.
 */
int AWG::load_data(int seg_num, short *p_data, uint64 size, bool wait_until_finished) {
    int dwSegLenByte = dwFactor * lSetChannels * size * bps; // Converting the number of samples to bytes for the total interleaved buffer.

    /// select segment to upload to
    spcm_dwSetParam_i32(p_card, SPC_SEQMODE_WRITESEGMENT, seg_num);
#ifdef ENABLE_CUDA
    spcm_dwDefTransfer_i64(p_card, SPCM_BUF_DATA, SPCM_DIR_GPUTOCARD, 0, p_data,
                           0, dwSegLenByte);
#else
    spcm_dwDefTransfer_i64(p_card, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, 0, p_data,
                           0, dwSegLenByte);
#endif

    return spcm_dwSetParam_i32(p_card, SPC_M2CMD,
                               M2CMD_DATA_STARTDMA | (wait_until_finished ? M2CMD_DATA_WAITDMA : 0));
}

/**
 * @brief: Waits for segment being written to finish if there is any
 */
int AWG::wait_for_data_load() {
    return spcm_dwSetParam_i32(p_card, SPC_M2CMD, M2CMD_DATA_WAITDMA);
}

/**
 * @brief initialize segment number seg_num with num_samples of samples
 * @return Error code
 */
int AWG::init_segment(int seg_num, int num_samples) {
    spcm_dwSetParam_i32(p_card, SPC_SEQMODE_WRITESEGMENT, seg_num);
    if (spcm_dwSetParam_i32(p_card, SPC_SEQMODE_SEGMENTSIZE, num_samples) !=
        ERR_OK) {
        std::cerr << "ERROR: AWG Sequence -- failed to initialize segment = "
                  << seg_num << std::endl;
        print_awg_error();
        return AWG_ERR;
    }
    return AWG_OK;
}

/**
 * @brief: initialize all AWG data memory segments and upload one given set of
 * samples to all of them
 * @param p_segment => pointer to array containing samples (16bit integers)
 * @param num_samples => number of samples from p_segment to load, starting from
 * first index
 * @return status code
 */
int AWG::init_and_load_all(short *p_segment, int num_samples) {
    int status;
    int dwSegLenSamples = dwFactor * num_samples * lSetChannels;
    for (int idx = 0; idx < config.awg_num_segments; idx++) {
        status |= init_segment(idx, num_samples);
        status |= load_data(idx, p_segment, dwSegLenSamples);
    }

    return status;
}

/**
 * @brief: initialize a range of the form [start, end) of AWG data memory
 * segments and upload one given set of samples to all of them
 * @param p_segment => pointer to array containing samples (16bit integers)
 * @param num_samples => number of samples from p_segment to load, starting from
 * first index
 * @param start => first segment idx to upload to
 * @param end => index write after last index to upload to
 * @return status code
 */
int AWG::init_and_load_range(short *p_segment, int num_samples, int start,
                             int end) {
    int status;
    int dwSegLenSamples = dwFactor * num_samples * lSetChannels;
    for (int idx = start; idx < end; idx++) {
        status |= init_segment(idx, num_samples);
        status |= load_data(idx, p_segment, dwSegLenSamples);
    }

    return status;
}

/**
 * @brief: Get current step that is streaming in the sequence memory of AWG
 * @return the step at which the error was generated, i.e. the step that is
 * streaming in the sequence memory
 */
int AWG::get_current_step() {
    int32 current_step = INT_MAX;
    if (spcm_dwGetParam_i32(p_card, SPC_SEQMODE_STATUS, &current_step) !=
        ERR_OK) {
        std::cerr << "ERROR: AWG Sequence -- failed to get current step."
                  << std::endl;
        print_awg_error();
    }

    return current_step;
}

/**
 * @brief Allocates a transfer buffer based on the number of samples
 * @param num_samples Number of samples
 * @param pnData Pointer to allocated transfer buffer
 * @return Size of allocated transfer buffer
 */
AWG::TransferBuffer AWG::allocate_transfer_buffer(int num_samples,
                                                  bool contBuf) {
    size_t qwBufferSize = lSetChannels * dwFactor * num_samples * bps;
    return TransferBuffer(*this, qwBufferSize,
                          (qwBufferSize <= continuousBufferSize) && contBuf);
}

/**
 * @brief Fills transfer buffer with a specified value
 * @param pnData Pointer to transfer buffer
 * @param num_samples Number of samples
 * @param value Value to fill the transfer buffer with
 * @return status code
 */
int AWG::fill_transfer_buffer(TransferBuffer &tb, int num_samples,
                              int16 value) {
    int dwSegmentLenSample = dwFactor * num_samples;
    for (int i = 0; i < dwSegmentLenSample; i++) {
        for (int lChannel = 0; lChannel < num_channels; ++lChannel) {
            ((short *)*tb)[i * num_channels + lChannel] = value;
        }
    }
    return 0;
}

/**
 * @brief Helper function to print error status of AWG card.
 */
void AWG::print_awg_error() {
    char error_text[ERRORTEXTLEN];
    spcm_dwGetErrorInfo_i32(p_card, NULL, NULL, error_text);
    std::cerr << error_text << std::endl;
}

/**
 * @brief Helper function to get error status of AWG card.
 */
std::tuple<int, std::string> AWG::get_awg_error() {
    char error_text[ERRORTEXTLEN];
    auto result = spcm_dwGetErrorInfo_i32(p_card, NULL, NULL, error_text);
    if (result != ERR_OK) {
        return {result, std::string(error_text)};
    } else {
        return {result, ""};
    }
}

AWG::TransferBuffer::TransferBuffer(AWG &awg, size_t size, bool contBuf)
    : buffer{nullptr}, size{size}, contBuf{contBuf} {
    if (contBuf) {
        buffer = awg.continuousBuffer;
        awg.continuousBuffer = awg.continuousBuffer + size;
        awg.continuousBufferSize -= size;
    } else {
        buffer = (void *)pvAllocMemPageAligned(size);
    }
}

AWG::TransferBuffer::TransferBuffer(AWG::TransferBuffer &&other) {
    std::swap(buffer, other.buffer);
    std::swap(size, other.size);
    std::swap(contBuf, other.contBuf);
}
AWG::TransferBuffer &
AWG::TransferBuffer::operator=(AWG::TransferBuffer &&other) {
    if (this != &other) {
        std::swap(buffer, other.buffer);
        std::swap(size, other.size);
        std::swap(contBuf, other.contBuf);
    }
    return *this;
}

AWG::TransferBuffer::~TransferBuffer() {
    if (!contBuf) {
        vFreeMemPageAligned(buffer, size);
    }
    /// The continuous buffer can not be freed, we can design our own allocation
    /// system for the buffer but it's not worth the effort as it will very
    /// rarely be useful.
}
