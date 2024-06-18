#include "WaveformTable.h"

/// Helper

std::vector<short> Synthesis::translate_waveform(const std::vector<double> &src,
                                                 int waveform_mask) {
    short new_max = waveform_mask;
    std::vector<short> sv;

    for (size_t i = 0; i < src.size(); ++i) {
        // assuming vector of doubles are normalized, sets new _primary_size
        short sample = short(round(src[i] * new_max));
        sv.push_back(sample);
    }

    return sv;
}

/// WaveformTable Class

Synthesis::WaveformTable::WaveformTable(WaveformRepo *p_repo,
                                        bool is_transposed, int waveform_mask) {

    this->is_transposed = is_transposed;
    this->waveform_mask = waveform_mask;
    this->secondary_chan = (is_transposed) ? CHAN_1 : CHAN_0;
    this->primary_chan = (is_transposed) ? CHAN_0 : CHAN_1;
    this->secondary_size = (is_transposed) ? (*p_repo).get_dimension_y()
                                           : (*p_repo).get_dimension_x();
    this->primary_size = (is_transposed) ? (*p_repo).get_dimension_x()
                                         : (*p_repo).get_dimension_y();
    this->primary_static = init_primary(STATIC, primary_size, p_repo);
    this->primary_extract = init_primary(EXTRACT, primary_size, p_repo);
    this->primary_implant = init_primary(IMPLANT, primary_size, p_repo);

    primary_forward.resize(primary_size + 1);
    for (int i = 2; i <= primary_size; ++i) {
        primary_forward[i] = init_primary(FORWARD, i - 1, p_repo);
    }
    primary_backward.resize(primary_size + 1);
    for (int i = 2; i <= primary_size; ++i) {
        primary_backward[i] = init_primary(BACKWARD, i - 1, p_repo);
    }

    secondary_static = init_secondary(STATIC, secondary_size, p_repo);
    secondary_extract = init_secondary(EXTRACT, secondary_size, p_repo);
    secondary_implant = init_secondary(IMPLANT, secondary_size, p_repo);
    secondary_rightward = init_secondary(RIGHTWARD, secondary_size - 1, p_repo);
    secondary_leftward = init_secondary(LEFTWARD, secondary_size - 1, p_repo);
    init_table();
}

std::vector<short> Synthesis::WaveformTable::interleave_waveforms(
    std::vector<short> primary_wf, std::vector<short> secondary_wf) {
    std::vector<short> result(primary_wf.size() + secondary_wf.size());

    double primary_coefficient = is_transposed ? 1 : 0.5;
    double secondary_coefficient = is_transposed ? 0.5 : 1;
    for (size_t idx = 0; idx < primary_wf.size() && idx < secondary_wf.size();
         idx++) {
        result[idx * 2 + primary_chan] = primary_wf[idx] * primary_coefficient;
        result[idx * 2 + secondary_chan] =
            secondary_wf[idx] * secondary_coefficient;
    }

    return std::move(result);
}

size_t Synthesis::WaveformTable::get_blocked_addr(WfType wf_type, size_t index,
                                                  size_t block_size,
                                                  size_t max) {
    int adjusted_index =
        (wf_type == BACKWARD) ? index - (primary_size - 1 - max) : index;
    // std::cout << index << " " << (_primary_size-1-max) << " " <<
    // adjusted_index << " " << max << std::endl;
    return adjusted_index * (max - adjusted_index) +
           (adjusted_index * (adjusted_index + 1)) / 2 + block_size - 1;
}

short *Synthesis::WaveformTable::get_primary_wf_ptr(WfType wf_type,
                                                    size_t index,
                                                    size_t block_size,
                                                    size_t extraction_extent) {
    WF_PAGE *wp = get_primary_pointer(wf_type, extraction_extent);

    size_t max = (wf_type == FORWARD || wf_type == BACKWARD)
                     ? extraction_extent - 1
                     : primary_size;
    return wp->at(get_blocked_addr(wf_type, index, block_size, max)).data();
}

std::vector<short> Synthesis::WaveformTable::get_secondary_wf(WfType wf_type,
                                                              size_t index) {
    WF_PAGE *p_offset = get_secondary_pointer(wf_type);
    return p_offset->at(index);
}

Synthesis::WF_PAGE *
Synthesis::WaveformTable::get_primary_pointer(WfType wf_type,
                                              size_t extraction_extent) {
    switch (wf_type) {
    case STATIC:
        return &primary_static;
    case EXTRACT:
        return &primary_extract;
    case IMPLANT:
        return &primary_implant;
    case FORWARD:
        return &primary_forward[extraction_extent];
    case BACKWARD:
        return &primary_backward[extraction_extent];
    default:
        throw std::invalid_argument("Invalid Primary Waveform Type");
    }
}

Synthesis::WF_PAGE *
Synthesis::WaveformTable::get_secondary_pointer(WfType wf_type) {
    switch (wf_type) {
    case STATIC:
        return &secondary_static;
    case EXTRACT:
        return &secondary_extract;
    case IMPLANT:
        return &secondary_implant;
    case RIGHTWARD:
        return &secondary_rightward;
    case LEFTWARD:
        return &secondary_leftward;
    default:
        throw std::invalid_argument("Invalid Secondary Waveform Type");
    }
}

/**
 * @brief Get a pointer to a specified waveform
 *
 * @param move  The Enum corresponding to the Wf's move
 * @param extraction_extent
 * @param index
 * @param offset
 * @param block_size
 * @return short* Pointer to the waveform
 */
short *Synthesis::WaveformTable::get_waveform_ptr(WfMoveType move,
                                                  int extraction_extent,
                                                  int index, int offset,
                                                  int block_size) {
    if (move == RIGHT_2D || move == LEFT_2D || move == IDLE_1D)
        extraction_extent = 0;
    TABLE_PAGE &tp = base_table.at(move + 2 * extraction_extent);
    WfType wf_type_primary = std::get<0>(tp);
    WfType wf_type_secondary = std::get<1>(tp);
    int max = std::get<2>(tp);

    size_t primary_block_addr =
        get_blocked_addr(wf_type_primary, index, block_size, max);
    size_t primary_size =
        get_primary_pointer(wf_type_primary, extraction_extent)->size();
    size_t block_address;
    if (move == FORWARD_1D || move == BACKWARD_1D || IMPLANT_1D || EXTRACT_1D ||
        IDLE_1D) {
        block_address = primary_size * offset + primary_block_addr;
    } else {
        block_address = primary_size * index + primary_block_addr;
    }
    return (std::get<3>(tp).at(block_address)).data();
}

void Synthesis::WaveformTable::init_table() {
    base_table.clear();
    base_table.resize(13 + 2 * primary_size);
    if (secondary_size == 1) { // 1D
        WF_PAGE *p_static = get_primary_pointer(STATIC, 0);
        WF_PAGE *p_implant = get_primary_pointer(IMPLANT, 0);
        WF_PAGE *p_extract = get_primary_pointer(EXTRACT, 0);

        base_table[IDLE_1D] =
            std::make_tuple(STATIC, NULL_WF, primary_size, *p_static);
        base_table[IMPLANT_1D] =
            std::make_tuple(IMPLANT, NULL_WF, primary_size, *p_implant);
        base_table[EXTRACT_1D] =
            std::make_tuple(EXTRACT, NULL_WF, primary_size, *p_extract);

        for (int extraction_extent = 2; extraction_extent <= primary_size;
             ++extraction_extent) {
            WF_PAGE *p_forward =
                get_primary_pointer(FORWARD, extraction_extent);
            WF_PAGE *p_backward =
                get_primary_pointer(BACKWARD, extraction_extent);
            int max = extraction_extent - 1;
            base_table[FORWARD_1D + 2 * extraction_extent] =
                std::make_tuple(FORWARD, NULL_WF, max, *p_forward);
            base_table[BACKWARD_1D + 2 * extraction_extent] =
                std::make_tuple(BACKWARD, NULL_WF, max, *p_backward);
        }
    } else { // 2D
        WF_PAGE *p_static = get_primary_pointer(STATIC, 0);
        WF_PAGE *p_implant = get_primary_pointer(IMPLANT, 0);
        WF_PAGE *p_extract = get_primary_pointer(EXTRACT, 0);
        WF_PAGE *s_static = get_secondary_pointer(STATIC);
        WF_PAGE *s_implant = get_secondary_pointer(IMPLANT);
        WF_PAGE *s_extract = get_secondary_pointer(EXTRACT);
        WF_PAGE *s_right = get_secondary_pointer(RIGHTWARD);
        WF_PAGE *s_left = get_secondary_pointer(LEFTWARD);

        base_table[IDLE_2D + 0] = std::make_tuple(
            STATIC, STATIC, primary_size, merge_pages(p_static, s_static));
        base_table[IMPLANT_2D + 0] = std::make_tuple(
            IMPLANT, IMPLANT, primary_size, merge_pages(p_implant, s_implant));
        base_table[EXTRACT_2D + 0] = std::make_tuple(
            EXTRACT, EXTRACT, primary_size, merge_pages(p_extract, s_extract));
        base_table[RIGHT_2D + 0] = std::make_tuple(
            STATIC, RIGHTWARD, secondary_size, merge_pages(p_static, s_right));
        base_table[LEFT_2D + 0] = std::make_tuple(
            STATIC, LEFTWARD, secondary_size, merge_pages(p_static, s_left));

        for (size_t extraction_extent = 2; extraction_extent <= primary_size;
             ++extraction_extent) {
            WF_PAGE *p_forward =
                get_primary_pointer(FORWARD, extraction_extent);
            WF_PAGE *p_backward =
                get_primary_pointer(BACKWARD, extraction_extent);
            size_t max = extraction_extent - 1;
            base_table[UP_2D + 2 * extraction_extent] = std::make_tuple(
                FORWARD, STATIC, max, merge_pages(p_forward, s_static));
            base_table[DOWN_2D + 2 * extraction_extent] = std::make_tuple(
                BACKWARD, STATIC, max, merge_pages(p_backward, s_static));
        }
    }
}

Synthesis::WF_PAGE
Synthesis::WaveformTable::init_primary(WfType wf_type, size_t range,
                                       Synthesis::WaveformRepo *p_repo) {
    WF_PAGE primary_table(range * (range + 1) / 2);
    int extraction_extent =
        (wf_type == FORWARD || wf_type == BACKWARD) ? range + 1 : 0;
    for (int adjusted_index = 0; adjusted_index < range; adjusted_index++) {
        int index = (wf_type == BACKWARD)
                        ? primary_size - 1 - range + adjusted_index
                        : adjusted_index;
        for (int block_size = 1; adjusted_index + block_size <= range;
             block_size++) {
            size_t address =
                get_blocked_addr(wf_type, index, block_size, range);
            primary_table[address] = translate_waveform(
                p_repo->get_waveform(primary_chan, wf_type, index, block_size,
                                     extraction_extent),
                waveform_mask);
        }
    }

    return primary_table;
}

Synthesis::WF_PAGE
Synthesis::WaveformTable::init_secondary(WfType wf_type, size_t range,
                                         Synthesis::WaveformRepo *p_repo) {
    WF_PAGE offset_table(range);
    for (int index = 0; index < range; index++) {
        offset_table[index] = translate_waveform(
            p_repo->get_waveform(secondary_chan, wf_type, index, 1, 0),
            waveform_mask);
    }
    return offset_table;
}

Synthesis::WF_PAGE Synthesis::WaveformTable::merge_pages(WF_PAGE *p_base,
                                                         WF_PAGE *p_offset) {
    WF_PAGE combined_page;
    for (int shft_idx = 0; shft_idx < p_offset->size(); shft_idx++) {
        for (int base_idx = 0; base_idx < p_base->size(); base_idx++) {
            combined_page.push_back(interleave_waveforms(
                p_base->at(base_idx), p_offset->at(shft_idx)));
        }
    }

    return combined_page;
}
