#include "synthesiser.h"

std::unordered_map<Synthesiser::Move, std::vector<short>,
                   Synthesiser::MoveHasher>
    Synthesiser::cache{};

void element_wise_add(std::vector<short> &dest, std::vector<short> src);
std::vector<short> translate(std::vector<double> src);

Synthesiser::Synthesiser(std::string coef_x_path, std::string coef_y_path) {
    Setup::read_fparams(coef_x_path, coef_x, 21);
    Setup::read_fparams(coef_y_path, coef_y, 1);
}

void Synthesiser::set_config(MovementsConfig movementsConfig) {
    moves.clear();
    moves.reserve(movementsConfig.get_movement_count());
    for (int i = 0; i < movementsConfig.get_movement_count(); i++) {
        moves.push_back(process_move(movementsConfig, i));
    }
}

Synthesiser::Move Synthesiser::process_move(MovementsConfig &movementsConfig,
                                            int i) {
    Synthesiser::Move move;
    auto moveConfig = movementsConfig.get_movement_waveforms().at(i);
    move.wait_for_trigger =
        (i == movementsConfig.get_movement_count() - 1)
            ? true
            : boost::get<long>(moveConfig.at("wait_on_trigger"));
    move.type = (WfType)boost::get<long>(moveConfig.at("move_type"));
    move.duration = boost::get<double>(moveConfig.at("duration"));
    move.index = boost::get<long>(moveConfig.at("index_x"));
    move.offset = boost::get<long>(moveConfig.at("index_y"));
    move.block_size = boost::get<long>(moveConfig.at("block_size"));
    move.func_type = (WFFuncType)boost::get<long>(moveConfig.at("wf_type"));
    move.extraction_extent =
        boost::get<long>(moveConfig.at("extraction_extent"));
    if (move.func_type == TANH_TRANSITION || move.func_type == ERF_TRANSITION) {
        move.vmax = boost::get<double>(moveConfig.at("vmax"));
    }
    return move;
}

std::vector<short> Synthesiser::synthesise(Move move) {
    // check the memoization table
    auto it = cache.find(move);
    if (it != cache.end()) {
        return it->second;
    }

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

    // Adding the fade in and fade out waveforms and extraction extent static
    // waveforms for forward/backwards waveforms
    double d_nu_1, d_nu_2;
    if (move.type == FORWARD || move.type == BACKWARD) {
        d_nu_1 = move.index > 1 ? std::get<1>(coef_x.at(move.index)) -
                                      std::get<1>(coef_x.at(move.index - 1))
                                : std::get<1>(coef_x.at(move.index + 1)) -
                                      std::get<1>(coef_x.at(move.index));
        d_nu_1 = d_nu_1 / 2;
        d_nu_2 =
            move.index + move.block_size < coef_x.size() - 1
                ? std::get<1>(coef_x.at(move.index + move.block_size + 1)) -
                      std::get<1>(coef_x.at(move.index + move.block_size))
                : std::get<1>(coef_x.at(coef_x.size() - 1)) -
                      std::get<1>(coef_x.at(coef_x.size() - 2));
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
                    translate(Synthesis::Idle(move.duration,
                                              std::make_tuple(alpha, nu, phi))
                                  .discretize(sample_rate)));
            }
            for (int i = move.index + move.block_size + 1;
                 i < move.extraction_extent; i++) {
                std::tie(alpha, nu, phi) = coef_x.at(i);
                element_wise_add(
                    wfm,
                    translate(Synthesis::Idle(move.duration,
                                              std::make_tuple(alpha, nu, phi))
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
            std::tie(alpha, nu, phi) =
                coef_x.at(move.index + move.block_size - 1);
            element_wise_add(
                wfm, translate(Synthesis::Displacement(
                                   move.duration,
                                   std::make_tuple((double)0, nu + d_nu_2, phi),
                                   std::make_tuple(alpha, nu, phi))
                                   .discretize(sample_rate)));

            for (unsigned int i = 0;
                 i < coef_x.size() - move.index - move.block_size; i++) {
                std::tie(alpha, nu, phi) = coef_x.at(coef_x.size() - 1 - i);
                element_wise_add(
                    wfm,
                    translate(Synthesis::Idle(move.duration,
                                              std::make_tuple(alpha, nu, phi))
                                  .discretize(sample_rate)));
            }
            for (int i = coef_x.size() + 1 - move.index;
                 i < move.extraction_extent; i++) {
                std::tie(alpha, nu, phi) = coef_x.at(coef_x.size() - 1 - i);
                element_wise_add(
                    wfm,
                    translate(Synthesis::Idle(move.duration,
                                              std::make_tuple(alpha, nu, phi))
                                  .discretize(sample_rate)));
            }
        }
    }

    // Adding the waveforms associated with the block
    for (int b = 0; b < move.block_size; ++b) {
        switch (move.type) {
        case NULL_WF:
            break;
        case STATIC:
            element_wise_add(
                wfm, translate(Synthesis::Idle(move.duration,
                                               coef_x.at(move.index + b))
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

    cache.insert({move, wfm});
    return wfm;
}

void Synthesiser::synthesise_and_upload(AWG &awg, int start_segment) {

    // init segments
    sample_rate = awg.get_sample_rate();
    for (unsigned int i = 0; i < moves.size(); i++) {
        AWG::TransferBuffer buffer = awg.allocate_transfer_buffer(
            static_cast<int>(moves[i].duration * sample_rate), false);
        std::vector<short> waveform = synthesise(moves[i]);
        memcpy(
            *buffer, waveform.data(),
            static_cast<int>(moves[i].duration * sample_rate * sizeof(short)));
        awg.init_and_load_range(
            *buffer, static_cast<int>(moves[i].duration * sample_rate),
            i + start_segment, i + 1 + start_segment);
    }

    // init steps
    awg.seqmem_update(start_segment - 1, start_segment - 1, 1,
                      start_segment - 1, SPCSEQ_ENDLOOPONTRIG);
    awg.force_hardware_trigger();
    awg.seqmem_update(start_segment - 1, start_segment - 1, 1, start_segment,
                      SPCSEQ_ENDLOOPONTRIG);

    int i;
    for (unsigned int iterator = 0; iterator < moves.size() - 1; iterator++) {
        i = iterator + start_segment;
        awg.seqmem_update(i, i, 1, i + 1,
                          moves.at(iterator).wait_for_trigger
                              ? SPCSEQ_ENDLOOPONTRIG
                              : SPCSEQ_ENDLOOPALWAYS);
    }

    awg.seqmem_update(start_segment + moves.size() - 1,
                      start_segment + moves.size() - 1, 1, start_segment - 1,
                      SPCSEQ_ENDLOOPONTRIG);
}

void element_wise_add(std::vector<short> &dest, std::vector<short> src) {
    assert(src.size() == dest.size());
    for (unsigned int i = 0; i < dest.size(); ++i) {
        dest[i] += src[i];
    }
}

std::vector<short> translate(std::vector<double> src) {
    std::vector<short> dest;
    dest.reserve(src.size());
    for (unsigned int i = 0; i < src.size(); i++) {
        assert(src.at(i) <= 1 && src.at(i) >= -1);
        dest.push_back((short)(src.at(i) * (0x00007fff)));
    }
    return dest;
}

void Synthesiser::reset(AWG &awg, int start_segment) {
    awg.seqmem_update(start_segment - 1, start_segment - 1, 1,
                      start_segment - 1, SPCSEQ_ENDLOOPALWAYS);

    int current_step = awg.get_current_step();
    if (current_step == start_segment - 1) {
        return;
    }

    std::cout << "Warning! The triggers do not restore the AWG steps to the "
                 "beginning, resulting in additional time wasted in fixing "
                 "this problem."
              << std::endl;
    int move_index = current_step - start_segment;
    while (!moves.at(move_index).wait_for_trigger) {
        move_index++;
    }

    awg.seqmem_update(move_index + start_segment, move_index + start_segment, 1,
                      start_segment - 1, SPCSEQ_ENDLOOPALWAYS);
    awg.force_hardware_trigger();

    while (awg.get_current_step() != start_segment - 1) {
    }
}

bool Synthesiser::Move::operator==(const Move &other) const {
    return wait_for_trigger == other.wait_for_trigger && type == other.type &&
           duration == other.duration && index == other.index &&
           offset == other.offset && block_size == other.block_size &&
           func_type == other.func_type &&
           extraction_extent == other.extraction_extent && vmax == other.vmax;
}

size_t Synthesiser::MoveHasher::operator()(const Move &move) const {
    return std::hash<int>()(move.type) ^ std::hash<int>()(move.index) ^
           std::hash<int>()(move.offset) ^ std::hash<int>()(move.block_size) ^
           std::hash<int>()(move.extraction_extent) ^
           std::hash<double>()(move.duration) ^
           std::hash<int>()(move.func_type) ^ std::hash<double>()(move.vmax) ^
           std::hash<bool>()(move.wait_for_trigger);
}