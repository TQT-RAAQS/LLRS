#include "synthesiser.h"
#include <fstream>

void element_wise_add(std::vector<short> &dest, std::vector<short> src);
std::vector<short> translate(std::vector<double> src);

Synthesiser::Synthesiser(std::string coef_x_path, std::string coef_y_path,
                         MovementsConfig movementsConfig) {
    for (int i = 0; i < movementsConfig.get_movement_count(); i++) {
        moves.push_back(process_move(movementsConfig, i));
    }
    Setup::read_fparams(coef_x_path, coef_x, 21);
    Setup::read_fparams(coef_y_path, coef_y, 1);
}

Synthesiser::Move Synthesiser::process_move(MovementsConfig movementsConfig,
                                            int i) {
    Synthesiser::Move move;
    auto moveConfig = movementsConfig.get_movement_waveforms().at(i);
    move.wait_for_trigger = movementsConfig.get_movement_triggers().at(i);
    move.type = (WfType)boost::get<long>(moveConfig.at("move_type"));
    move.duration = boost::get<double>(moveConfig.at("duration"));
    move.index = boost::get<long>(moveConfig.at("index_x"));
    move.offset = boost::get<long>(moveConfig.at("index_y"));
    move.block_size = boost::get<long>(moveConfig.at("block_size"));
    move.func_type = (WFFuncType)boost::get<long>(moveConfig.at("wf_type"));
    move.extraction_extent = boost::get<long>(moveConfig.at("extraction_extent"));
    if (move.func_type == TANH_TRANSITION || move.func_type == ERF_TRANSITION) {
        move.vmax = boost::get<double>(moveConfig.at("vmax"));
    }
    return move;
}

std::vector<short> Synthesiser::synthesise(Move move) {
    // Specifying the mathematical function generating the waveforms
    switch (move.func_type) {
    case SIN_STATIC:
        Synthesis::Waveform::set_static_function(
            std::make_unique<Synthesis::Sin>());
        break;
    case ERF_TRANSITION:
        Synthesis::Waveform::set_transition_function(
            std::make_unique<Synthesis::ERF>(move.duration, move.vmax));
        Synthesis::Waveform::set_static_function(
            std::make_unique<Synthesis::Sin>());
        break;
    case TANH_TRANSITION:
        Synthesis::Waveform::set_transition_function(
            std::make_unique<Synthesis::TANH>(move.duration, move.vmax));
        Synthesis::Waveform::set_static_function(
            std::make_unique<Synthesis::Sin>());
        break;
    case SPLINE_TRANSITION:
        Synthesis::Waveform::set_transition_function(
            std::make_unique<Synthesis::Spline>(move.duration));
        Synthesis::Waveform::set_static_function(
            std::make_unique<Synthesis::Sin>());
        break;
    }

    std::vector<short> wfm(static_cast<int>(move.duration * sample_rate), 0);

    // Adding the fade in and fade out waveforms and extraction extent static waveforms
    // for forward/backwards waveforms
    double d_nu_1, d_nu_2;
    if (move.type == FORWARD || move.type == BACKWARD) {
        d_nu_1 = move.index > 1 ? std::get<0>(coef_x.at(move.index)) -
                                  std::get<0>(coef_x.at(move.index - 1))
                            : std::get<0>(coef_x.at(move.index + 1)) -
                                  std::get<0>(coef_x.at(move.index));
        d_nu_1 = d_nu_1 / 2;
        d_nu_2 =
            move.index + move.block_size != coef_x.size() - 1
                ? std::get<0>(coef_x.at(move.index + move.block_size + 1)) -
                      std::get<0>(coef_x.at(move.index + move.block_size))
                : std::get<0>(coef_x.at(move.index + move.block_size)) -
                      std::get<0>(coef_x.at(move.index + move.block_size - 1));
        d_nu_2 = d_nu_2 / 2;
        if (move.type == FORWARD) {
            assert(move.index + move.block_size < coef_x.size());
            double alpha, nu, phi;
            std::tie(alpha, nu, phi) = coef_x.at(move.index);
            element_wise_add(
                wfm, translate(Synthesis::Displacement(
                                   move.duration,
                                   std::make_tuple((double)0, nu - d_nu_1, phi),
                                   std::make_tuple(alpha, nu, phi))
                                   .discretize(sample_rate)));
            std::tie(alpha, nu, phi) = coef_x.at(move.index + move.block_size);
            element_wise_add(
                wfm,
                translate(Synthesis::Displacement(
                              move.duration, std::make_tuple(alpha, nu, phi),
                              std::make_tuple((double)0, nu + d_nu_2, phi))
                              .discretize(sample_rate)));

            for (int i = 0; i < move.index; i++) {
                std::tie(alpha, nu, phi) = coef_x.at(i);
                element_wise_add(
                wfm,
                translate(Synthesis::Idle(
                              move.duration, std::make_tuple(alpha, nu, phi))
                              .discretize(sample_rate)));
            }
            for (int i = move.index + move.block_size + 1; i < move.extraction_extent; i++) {
                std::tie(alpha, nu, phi) = coef_x.at(i);
                element_wise_add(
                wfm,
                translate(Synthesis::Idle(
                              move.duration, std::make_tuple(alpha, nu, phi))
                              .discretize(sample_rate)));
            }
        } else {
            assert(move.index > 0);
            double alpha, nu, phi;
            std::tie(alpha, nu, phi) = coef_x.at(move.index - 1);
            element_wise_add(
                wfm,
                translate(Synthesis::Displacement(
                              move.duration, std::make_tuple(alpha, nu, phi),
                              std::make_tuple((double)0, nu - d_nu_1, phi))
                              .discretize(sample_rate)));
            std::tie(alpha, nu, phi) = coef_x.at(move.index + move.block_size - 1);
            element_wise_add(
                wfm, translate(Synthesis::Displacement(
                                   move.duration,
                                   std::make_tuple((double)0, nu + d_nu_2, phi),
                                   std::make_tuple(alpha, nu, phi))
                                   .discretize(sample_rate)));

            for (int i = 0; 
                 i < coef_x.size() - move.index - move.block_size; 
                 i++) {
                std::tie(alpha, nu, phi) = coef_x.at(coef_x.size() - 1 - i);
                element_wise_add(
                wfm,
                translate(Synthesis::Idle(
                              move.duration, std::make_tuple(alpha, nu, phi))
                              .discretize(sample_rate)));
            }
            for (int i = coef_x.size() + 1 - move.index; 
                i < move.extraction_extent; 
                i++) {
                std::tie(alpha, nu, phi) = coef_x.at(coef_x.size() - 1 - i);
                element_wise_add(
                wfm,
                translate(Synthesis::Idle(
                              move.duration, std::make_tuple(alpha, nu, phi))
                              .discretize(sample_rate)));
            }
        }
    }

    // Adding the waveforms associated with the block
    for (int b = 0; b < move.block_size; ++b) {
        switch (move.type) {
        case STATIC:
            element_wise_add(
                wfm,
                translate(Synthesis::Idle(move.duration, coef_x.at(move.index + b))
                              .discretize(sample_rate)));
            break;
        case EXTRACT:
            throw std::invalid_argument("Extract not implemented.");
            break;
        case IMPLANT:
            throw std::invalid_argument("Implant not implemented.");
            break;
        case FORWARD:
            element_wise_add(
                wfm, translate(Synthesis::Displacement(
                                   move.duration, coef_x.at(move.index + b),
                                   coef_x.at(move.index + b + 1))
                                   .discretize(sample_rate)));
            break;
        case BACKWARD:
            element_wise_add(
                wfm, translate(Synthesis::Displacement(
                                   move.duration, coef_x.at(move.index + b),
                                   coef_x.at(move.index + b - 1))
                                   .discretize(sample_rate)));
            break;
        case RIGHTWARD:
            throw std::invalid_argument("Rightward not implemented.");
            break;
        case LEFTWARD:
            throw std::invalid_argument("Leftward not implemented.");
            break;
        }
    }

    return wfm;
}

void Synthesiser::synthesise_and_upload(AWG &awg, int start_segment) {

    // init segments
    sample_rate = awg.get_sample_rate();
    for (int i = 0; i < moves.size(); i++) {
        AWG::TransferBuffer buffer = awg.allocate_transfer_buffer(
            static_cast<int>(moves[i].duration * sample_rate), false);
        std::vector<short> waveform = synthesise(moves[i]);
        memcpy(
            *buffer, waveform.data(),
            static_cast<int>(moves[i].duration * sample_rate * sizeof(short)));
        awg.init_and_load_range(
            *buffer, static_cast<int>(moves[i].duration * sample_rate), i + start_segment,
            i + 1 + start_segment);
    }

    // init steps
    for (int i = start_segment; i < moves.size(); i++) {
        int j = i - 1 == -1 ? moves.size() - 1 : i - 1;
        if (moves[i].wait_for_trigger) {
            awg.seqmem_update(j, j, 1, i, SPCSEQ_ENDLOOPONTRIG);
        } else {
            awg.seqmem_update(j, j, 1, i, SPCSEQ_ENDLOOPALWAYS);
        }
    }
    awg.seqmem_update(start_segment + moves.size() - 1, start_segment - 1, 1, start_segment + moves.size() - 1, SPCSEQ_ENDLOOPALWAYS);
}

void element_wise_add(std::vector<short> &dest, std::vector<short> src) {
    assert(src.size() == dest.size());
    for (int i = 0; i < dest.size(); ++i) {
        dest[i] += src[i];
    }
}

std::vector<short> translate(std::vector<double> src) {
    std::vector<short> dest;
    dest.reserve(src.size());
    for (int i = 0; i < src.size(); i++) {
        assert (src.at(i) <= 1 && src.at(i) >= -1);
        dest.push_back((short)(src.at(i) * (0x00007fff)));
    }
    return dest;
}
