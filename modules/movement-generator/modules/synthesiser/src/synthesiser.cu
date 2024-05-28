#include "synthesiser.h"

void element_wise_add(std::vector<short> dest, std::vector<short> src);
std::vector<short> translate(std::vector<double> src);

Synthesiser::Synthesiser(std::string coef_x_path, std::string coef_y_path,
                         MovementsConfig movementsConfig) {
    for (int i = 0; i < movementsConfig.get_movement_count(); i++) {
        moves.push_back(process_move(movementsConfig, i));
    }
    Setup::read_fparams(coef_x_path, coef_x, 100);
    Setup::read_fparams(coef_y_path, coef_y, 1);
}

Synthesiser::Move Synthesiser::process_move(MovementsConfig movementsConfig,
                                            int i) {
    Synthesiser::Move move;
    auto moveConfig = movementsConfig.get_movement_waveforms().at(i);
    move.wait_for_trigger = movementsConfig.get_movement_triggers().at(i);
    move.type = (WfType)boost::get<double>(moveConfig.at("type"));
    move.duration = boost::get<double>(moveConfig.at("duration"));
    move.index = boost::get<double>(moveConfig.at("index"));
    move.offset = boost::get<double>(moveConfig.at("offset"));
    move.block_size = boost::get<double>(moveConfig.at("block_size"));
    move.func_type = (WFFuncType)boost::get<double>(moveConfig.at("func_type"));
    if (move.func_type == TANH_TRANSITION || move.func_type == ERF_TRANSITION) {
        move.vmax = boost::get<double>(moveConfig.at("vmax"));
    }
    return move;
}

std::vector<short> Synthesiser::synthesise(Move move) {
    switch (move.func_type) {
    case SIN_STATIC:
        Synthesis::Waveform::set_static_function(
            std::make_unique<Synthesis::Sin>());
        break;
    case ERF_TRANSITION:
        Synthesis::Waveform::set_transition_function(
            std::make_unique<Synthesis::ERF>(move.duration, move.vmax));
        break;
    case TANH_TRANSITION:
        Synthesis::Waveform::set_transition_function(
            std::make_unique<Synthesis::TANH>(move.duration, move.vmax));
        break;
    case SPLINE_TRANSITION:
        Synthesis::Waveform::set_transition_function(
            std::make_unique<Synthesis::Spline>(move.duration));
        break;
    }
    std::vector<short> wfm(static_cast<int>(move.duration * sample_rate), 0);
    double d_nu_1, d_nu_2;
    if (move.type == FORWARD || move.type == BACKWARD) {
        d_nu_1 = move.index ? std::get<0>(coef_x.at(move.index)) -
                                  std::get<0>(coef_x.at(move.index - 1))
                            : std::get<0>(coef_x.at(move.index + 1)) -
                                  std::get<0>(coef_x.at(move.index));
        d_nu_1 = d_nu_1 / 2;
        d_nu_2 =
            move.index + move.block_size != coef_x.size()
                ? std::get<0>(coef_x.at(move.index + move.block_size + 1)) -
                      std::get<0>(coef_x.at(move.index + move.block_size))
                : std::get<0>(coef_x.at(move.index + move.block_size)) -
                      std::get<0>(coef_x.at(move.index + move.block_size - 1));
        d_nu_2 = d_nu_2 / 2;
        if (move.type == FORWARD) {
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
        } else {
            double alpha, nu, phi;
            std::tie(alpha, nu, phi) = coef_x.at(move.index);
            element_wise_add(
                wfm,
                translate(Synthesis::Displacement(
                              move.duration, std::make_tuple(alpha, nu, phi),
                              std::make_tuple((double)0, nu - d_nu_1, phi))
                              .discretize(sample_rate)));
            std::tie(alpha, nu, phi) = coef_x.at(move.index + move.block_size);
            element_wise_add(
                wfm, translate(Synthesis::Displacement(
                                   move.duration,
                                   std::make_tuple((double)0, nu + d_nu_2, phi),
                                   std::make_tuple(alpha, nu, phi))
                                   .discretize(sample_rate)));
        }
    }

    for (int b = 0; b < move.block_size; ++b) {
        switch (move.type) {
        case STATIC:
            element_wise_add(
                wfm,
                translate(Synthesis::Idle(move.duration, coef_x.at(move.index))
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

void Synthesiser::synthesise_and_upload(AWG &awg) {

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
            *buffer, static_cast<int>(moves[i].duration * sample_rate), i,
            i + 1);
    }

    // init steps
    for (int i = 0; i < moves.size(); i++) {
        int j = i - 1 == -1 ? moves.size() - 1 : i - 1;
        if (moves[i].wait_for_trigger) {
            awg.seqmem_update(j, j, 1, i, SPCSEQ_ENDLOOPONTRIG);
        } else {
            awg.seqmem_update(j, j, 1, i, SPCSEQ_ENDLOOPALWAYS);
        }
    }
}

void element_wise_add(std::vector<short> dest, std::vector<short> src) {
    assert(src.size() == dest.size());
    for (int i = 0; i < dest.size(); ++i) {
        dest[i] += src[i];
    }
}

std::vector<short> translate(std::vector<double> src) {
    std::vector<short> dest;
    dest.reserve(src.size());
    for (int i = 0; i < src.size(); i++) {
        dest.push_back((short)(src.at(i) * (2 ^ 15 - 1)));
    }
    return dest;
}