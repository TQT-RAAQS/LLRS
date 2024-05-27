#include "synthesiser.h"

Synthesiser(std::string coef_x_path, std::string coef_y_path,std::vector<Move> moves): moves(moves) {
    Setup::read_fparams(coef_x_path, coef_x);
    Setup::read_fparams(coef_y_path, coef_y);
}
Synthesiser(std::string coef_x_path, std::string coef_y_path, std::vector<std::map> maps) {
    for (auto map : maps) {
        moves.push_back(process_hashmap(map));
    }
    Setup::read_fparams(coef_x_path, coef_x);
    Setup::read_fparams(coef_y_path, coef_y);
}


Move Synthesiser::process_hashmap(std::map<> hashmap) {
    // Fill here 
}


std::vector<short> Synthesiser::synthesise(Move move) {
    // Fill here
}

void Synthesiser::synthesise_and_upload(AWG &awg) {

    // init segments
    double sample_rate = awg->get_sample_rate();
    for (int i = 0; i < moves.size(); i++){
        AWG::TransferBuffer buffer = awg.allocate_transfer_buffer(std::static_cast<int>(moves[i].duration * sample_rate), false);
        std::vector<short> waveform = synthesise(moves[i]);
        memcpy(*buffer, waveform.data(), std::static_cast<int>(moves[i].duration * sample_rate * sizeof(short)));
        awg.init_and_load_range(*buffer, std::static_cast<int>(moves[i].duration * sample_rate), i, i + 1);
        
    } 

    // init steps
    for (int i = 0; i < moves.size(); i++){
        int j = i-1 == -1? moves.size() - 1 : i-1;
        if (moves[i].wait_for_trigger){
            awg.seqmem_update(j, j, 1, i, SPCSEQ_ENDLOOPONTRIG);
        } else {
            awg.seqmem_update(j, j, 1, i, SPCSEQ_ENDLOOPALWAYS);
        }
    }
}