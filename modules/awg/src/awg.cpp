/**
 * @brief Handles configurations for the AWG class
 * @date Dec 2023
 */

#include "awg.hpp"
#include <bits/stdc++.h>

/**
 * @brief Constructor for AWG class
 */
AWG::AWG() { configure(); }

/**
 * @brief Destructor for AWG class
 */
AWG::~AWG() {
    stop_card();
    close_card();
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
    std::string temp_driver_path = node["driver_path"].as<std::string>();
    config.driver_path = new char[temp_driver_path.size() + 1];
    std::strcpy(config.driver_path, temp_driver_path.c_str());
    config.external_clock_freq = node["external_clock_freq"].as<int>();
    config.channels = node["channels"].as<std::vector<int>>();
    config.amp = node["amp"].as<std::vector<int>>();
    config.awg_num_segments = node["awg_num_segments"].as<int>();
    config.sample_rate = node["sample_rate"].as<double>();
    std::string wfm_mask_str = node["wfm_mask"].as<std::string>();
    config.wfm_mask = std::stoi(wfm_mask_str, 0, 16);
    config.waveform_duration = node["waveform_duration"].as<double>();
    config.waveforms_per_segment = node["waveforms_per_segment"].as<int>();
    config.null_segment_length = node["null_segment_length"].as<int>();
    config.idle_segment_length = node["idle_segment_length"].as<int>();
    config.waveform_length =
        static_cast<int>(config.sample_rate * config.waveform_duration);
    config.samples_per_segment =
        config.waveforms_per_segment * config.waveform_length;
    config.trigger_size = node["trigger_size"].as<int>();
    config.vpp = node["vpp"].as<int>();
    config.acq_timeout = node["acq_timeout"].as<int>();
    config.async_trig_amp = node["async_trig_amp"].as<int>();
    config.idle_segment_wfm = node["idle_segment_wfm"].as<bool>();

    return 0;
}

/**
 * @brief Configure the AWG
 * @return Error code
 */
int AWG::configure() {
    if (read_config(AWG_CONFIG_PATH) != AWG_OK) {
        std::cerr << "Error occured in parsing AWG config.\n";
        return 1;
    }

    p_card = spcm_hOpen(config.driver_path);
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
    status |= set_external_clock_mode(config.external_clock_freq);
    status |= set_trigger_settings();
    status |= set_dout_async(SPCM_X0_MODE); /// x0 - null counter
    status |=
        set_dout_trigger_mode(SPCM_X1_MODE,
                              SPCM_XMODE_DIGOUTSRC_CH1); /// x1 - take image
    status |= set_dout_async(SPCM_X2_MODE); /// x2 - done/resume clock

    status |= spcm_dwGetParam_i32(p_card, SPC_SEQMODE_AVAILMAXSTEPS, &max_step);
    status |= spcm_dwGetParam_i32(p_card, SPC_MIINST_BYTESPERSAMPLE, &bps);
    status |= spcm_dwGetParam_i32(p_card, SPC_CHCOUNT, &lSetChannels);
    dwFactor = 1;

    status |= spcm_dwSetParam_i32(p_card, SPC_CARDMODE, SPC_REP_STD_SEQUENCE);
    status |= spcm_dwSetParam_i32(p_card, SPC_SEQMODE_MAXSEGMENTS,
                                  config.awg_num_segments);
    status |= spcm_dwSetParam_i32(p_card, SPC_SEQMODE_STARTSTEP, 0);

    if (config.awg_num_segments == 0 ||
        ((config.awg_num_segments & (config.awg_num_segments - 1)) != 0) ||
        config.awg_num_segments > max_step) {
        std::cerr << "ERROR: AWG Constructor -> max_seg needs to be "
                     "a power of 2 and less than "
                  << max_step << std::endl;
        return 1;
    }
    assert(config.samples_per_segment <=
           (AWG_MEMORY_SIZE / bps) /
               config.awg_num_segments); // assert that the samples per segments
                                         // matches the size of the AWG memory

    /// Allocate the continuous buffer
    spcm_dwGetContBuf_i64(p_card, SPCM_BUF_DATA, &continuousBuffer,
                          &continuousBufferSize);
    std::cout << "Physically continuous buffer of size " << continuousBufferSize 
              << " was successfully allocated." << std::endl;

    return status;
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
int AWG::set_trigger_settings() {
    int status = 0;
    status |= spcm_dwSetParam_i32(p_card, SPC_TRIG_EXT0_LEVEL0, 2000);
    status |= spcm_dwSetParam_i32(p_card, SPC_TRIG_EXT0_LEVEL1, 500);
    status |= spcm_dwSetParam_i32(p_card, SPC_TRIG_EXT0_MODE,
                                  SPC_TM_NEG | SPC_TM_REARM);
    status |= spcm_dwSetParam_i32(p_card, SPC_TRIG_ORMASK, SPC_TMASK_EXT0);
    status |= spcm_dwSetParam_i32(p_card, SPC_TIMEOUT, 5000);
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
int AWG::set_dout_async(int32 line) {
    return spcm_dwSetParam_i32(p_card, line, SPCM_XMODE_ASYNCOUT);
}

/**
 * @brief Async pulse is generated on all MPIOs that are set as
 * SPMCM_XMODE_ASYNCOUT
 * @return Error code
 */
void AWG::generate_async_output_pulse(TriggerType type) {
    switch (type) {
    case EMCCD:
        spcm_dwSetParam_i32(p_card, SPCM_XX_ASYNCIO, 0);
        spcm_dwSetParam_i32(p_card, SPCM_XX_ASYNCIO, config.async_trig_amp);
        spcm_dwSetParam_i32(p_card, SPCM_XX_ASYNCIO, 0);
    }
}

/**
 * @brief Software trigger on card start
 * @return Error code
 */
int AWG::start_stream() {
    return spcm_dwSetParam_i32(p_card, SPC_M2CMD,
                               M2CMD_CARD_START | M2CMD_CARD_FORCETRIGGER);
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
void AWG::close_card() { spcm_vClose(p_card); }

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

/**
 * @brief: Write to segment and returns status
 * @param seg_num => sequence segment number to refer to
 * @param p_data => pointer toa source data
 * @param size => size in bytes to write
 */
int AWG::load_data(int seg_num, short *p_data, uint64 size) {
    /// select segment to upload to
    spcm_dwSetParam_i32(p_card, SPC_SEQMODE_WRITESEGMENT, seg_num);
#ifdef ENABLE_CUDA
    spcm_dwDefTransfer_i64(p_card, SPCM_BUF_DATA, SPCM_DIR_GPUTOCARD, 0, p_data,
                           0, size);
#else
    spcm_dwDefTransfer_i64(p_card, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, 0, p_data,
                           0, size);
#endif

    return spcm_dwSetParam_i32(p_card, SPC_M2CMD,
                               M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA);
}

/**
 * @brief: Asynchronously start writing to a segment, returns status
 * @param seg_num => sequence segment number to refer to
 * @param p_data => pointer toa source data
 * @param size => size in bytes to write
 */
int AWG::load_data_async_start(int seg_num, short *p_data, uint64 size) {

    int dwSegLenByte = dwFactor * lSetChannels * size;
    /// select segment to upload to
    spcm_dwSetParam_i32(p_card, SPC_SEQMODE_WRITESEGMENT, seg_num);
    /// Transfer of data from PC memory to on-board memory of the card
#ifdef ENABLE_CUDA
    spcm_dwDefTransfer_i64(p_card, SPCM_BUF_DATA, SPCM_DIR_GPUTOCARD, 0, p_data,
                           0, size);
#else
    spcm_dwDefTransfer_i64(
        p_card, SPCM_BUF_DATA, SPCM_DIR_PCTOCARD, 0, p_data, 0,
        dwSegLenByte); // DKEA: changed this, need to verify correctness
#endif

    return spcm_dwSetParam_i32(p_card, SPC_M2CMD, M2CMD_DATA_STARTDMA);
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
    int dwSegLenByte = bps * dwFactor * num_samples * lSetChannels;
    for (int idx = 0; idx < config.awg_num_segments; idx++) {
        status |= init_segment(idx, num_samples);
        status |= load_data(idx, p_segment, dwSegLenByte);
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
    int dwSegLenByte = bps * dwFactor * num_samples * lSetChannels;
    for (int idx = start; idx < end; idx++) {
        status |= init_segment(idx, num_samples);
        status |= load_data(idx, p_segment, dwSegLenByte);
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
        for (int lChannel = 0; lChannel < lSetChannels; ++lChannel) {
            ((short *)*tb)[i * lSetChannels + lChannel] = value;
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

AWG::TransferBuffer::~TransferBuffer() {
    if (!contBuf) {
        vFreeMemPageAligned(buffer, size);
    }
    /// The continuous buffer can not be freed, we can design our own allocation
    /// system for the buffer but it's not worth the effort as it will very
    /// rarely be useful.
}
