#include "fsm.hpp"
using json = nlohmann::json;

/**
 * @brief Constructor of the Finite State Machine class
 *
 * Construct state objects for all states in the FSM. Connect the state
 * transitions between these states.
 */

FiniteStateMachine::FiniteStateMachine()
    : server_handler{}, trigger_detector{}, llrs{trigger_detector.getAWG()} {

    // 1. Create the ST_FAULT state
    State *fault_state =
        new State([this]() { this->st_FAULT(); }, []() { return -1; }, NULL);
    states.insert({ST_FAULT, fault_state});

    // Create dynamic state action function map
    dyn_state_action_func_map = {{M_LLRS, [this]() { this->st_LLRS_EXEC(); }}};

    // Configure static states of the FSM
    setupFSM();

    // Begin the state machine the in the ST_IDLE state
    currentState = states[ST_BEGIN];
}

/**
 * @brief The destructor of the Finite State Machine class.
 *
 * For every dynamically allocated state object, free the object.
 */
FiniteStateMachine::~FiniteStateMachine() {
    states.clear();
    for (auto stateVector : programmable_states) {
        for (State *state : stateVector) {
            delete state;
        }
        stateVector.clear();
    }
    programmable_states.clear();
}

/**
 * @brief The member function that creates all state objects for the states of
 * the non configurable part of the FSM .
 *
 * Adds this state object to the list of state objects in the fsm.
 *
 */
void FiniteStateMachine::setupFSM() {

    State *f = states[ST_FAULT];

    State *begin =
        new State([this]() { this->st_BEGIN(); }, []() { return 1; }, f);
    State *idle_state =
        new State([this]() { this->st_IDLE(); },
                  [this]() { return server_handler.get_request(); }, f);
    State *psf_reset_state =
        new State([this]() { this->st_CONFIG_PSF(); }, []() { return 1; }, f);
    State *waveform_reset_state = new State(
        [this]() { this->st_CONFIG_WAVEFORM(); }, []() { return 1; }, f);
    State *llrs_reset_state =
        new State([this]() { this->st_RESET(); }, []() { return 1; }, f);
    State *close_awg_state =
        new State([this]() { this->st_CLOSE_AWG(); },
                  [this]() { return server_handler.get_request(); }, f);
    State *restart_awg_state =
        new State([this]() { this->st_RESTART_AWG(); }, []() { return 1; }, f);
    State *process_shot_state =
        new State([this]() { this->st_PROCESS_SHOT(); }, []() { return 1; }, f);
    State *ready_state = new State(
        [this]() { this->st_READY(); },
        [this]() {
            return trigger_detector.detectTrigger(6000) != NO_HW_TRIG ? 1 : -1;
        },
        f);
    State *llrs_state = new State([this]() { this->st_LLRS_EXEC(); },
                                  [this]() { return 1; }, f);
    State *trigger_done_state = new State(
        [this]() { this->st_TRIGGER_DONE(); },
        [this]() { return this->commands_itr == commands.size() ? 1 : 2; }, f);
    State *exit_state =
        new State([this]() { this->st_EXIT(); }, []() { return 1; }, NULL);

    states.insert({ST_BEGIN, begin});
    states.insert({ST_IDLE, idle_state});
    states.insert({ST_RESET, llrs_reset_state});
    states.insert({ST_CLOSE_AWG, close_awg_state});
    states.insert({ST_RESTART_AWG, restart_awg_state});
    states.insert({ST_CONFIG_PSF, psf_reset_state});
    states.insert({ST_CONFIG_WAVEFORM, waveform_reset_state});
    states.insert({ST_PROCESS_SHOT, process_shot_state});
    states.insert({ST_READY, ready_state});
    states.insert({ST_LLRS_EXEC, llrs_state});
    states.insert({ST_TRIGGER_DONE, trigger_done_state});
    states.insert({ST_EXIT, exit_state});

    // Configure static transitions
    begin->addStateTransition(1, idle_state);

    idle_state->addStateTransition(1, process_shot_state);
    idle_state->addStateTransition(2, llrs_reset_state);
    idle_state->addStateTransition(3, psf_reset_state);
    idle_state->addStateTransition(4, waveform_reset_state);
    idle_state->addStateTransition(5, close_awg_state);

    close_awg_state->addStateTransition(5, restart_awg_state);
    restart_awg_state->addStateTransition(1, idle_state);

    process_shot_state->addStateTransition(1, ready_state);
    llrs_reset_state->addStateTransition(1, idle_state);
    psf_reset_state->addStateTransition(1, idle_state);
    waveform_reset_state->addStateTransition(1, idle_state);

    ready_state->addStateTransition(1, llrs_state);
    llrs_state->addStateTransition(1, trigger_done_state);

    trigger_done_state->addStateTransition(1, idle_state);
    trigger_done_state->addStateTransition(2, ready_state);

    idle_state->setType(ST_IDLE);
    begin->setType(ST_BEGIN);
    llrs_reset_state->setType(ST_RESET);
    psf_reset_state->setType(ST_CONFIG_PSF);
    waveform_reset_state->setType(ST_CONFIG_WAVEFORM);
    close_awg_state->setType(ST_CLOSE_AWG);
    restart_awg_state->setType(ST_RESTART_AWG);
    process_shot_state->setType(ST_PROCESS_SHOT);
    ready_state->setType(ST_READY);
    trigger_done_state->setType(ST_TRIGGER_DONE);
    exit_state->setType(ST_EXIT);
    f->setType(ST_FAULT);
}

void FiniteStateMachine::resetTransitions() {

    State *ready_state = states[ST_READY];
    State *fault_state = states[ST_FAULT];

    for (auto &stateVector : programmable_states) {
        for (State *state : stateVector) {
            delete state;
        }
        stateVector.clear();
    }
    programmable_states.clear();

    ready_state->resetTransitionMap();
}

void FiniteStateMachine::printStates() {
    for (const auto &pair : states) {
        State *state = pair.second;
        state->printState();
        std::cout << std::endl;
    }
}

void FiniteStateMachine::runFSM() {
    server_handler.start_listening();
    while (currentState != nullptr) {
        currentState->executeState();
        currentState = currentState->getNextState();
    }
}

/* * * * * * * * Static State definitions * * * * * * * */

void FiniteStateMachine::st_BEGIN() {
    std::cout << "FSM:: BEGIN state" << std::endl;
    trigger_detector.reset_segment_size();
    llrs.setup(
        llrs_problem_path, false,
        1); // Default problem, operator can reset the problem in the idle step

    std::cout << "Starting AWG stream" << std::endl;
    auto awg = trigger_detector.getAWG();
    auto tb = awg->allocate_transfer_buffer(
        trigger_detector.get_samples_per_idle_segment());

    if (awg->get_idle_segment_wfm()) {
        llrs.get_idle_wfm(tb, trigger_detector.get_samples_per_idle_segment());
    } else {
        awg->fill_transfer_buffer(
            tb, trigger_detector.get_samples_per_idle_segment(), 0);
    }
    trigger_detector.setup(tb);
    trigger_detector.stream();
}

void FiniteStateMachine::st_IDLE() {
    std::cout << "FSM:: IDLE state" << std::endl;
}

void FiniteStateMachine::st_PROCESS_SHOT() {
    std::cout << "FSM:: PROCESS_SHOT state" << std::endl;

    // receive hdf5 filepath from the workstation
    LLCSConfig llrs_config(ShotFile(server_handler.get_hdf5_file_path()));
    commands = llrs_config.get_commands();
    llrs_metadata.reserve(commands.size());
    commands_itr = 0;

    // Setting the LLRS problem for the first shot
    auto config = boost::get<LLRSCommandData>(commands[commands_itr].data);
    llrs.reset_problem(boost::get<std::string>(config.at("algorithm")),
                       boost::get<int>(config.at("T_x")) *
                           boost::get<int>(config.at("T_y")));

    server_handler.send_done();
}

void FiniteStateMachine::st_READY() {
    std::cout << "FSM:: READY state" << std::endl;
    std::cout << "Awaiting Hardware Trigger..." << std::endl;
}

void FiniteStateMachine::st_TRIGGER_DONE() {
    std::cout << "FSM:: TRIGGER_DONE state" << std::endl;
    ++commands_itr;
    if (commands_itr == commands.size()) {
        saveMetadata(server_handler.get_hdf5_file_path());
        llrs_metadata.clear();
        server_handler.wait_for_done();
        server_handler.send_done();
    } else {
        auto config = boost::get<LLRSCommandData>(commands[commands_itr].data);
        llrs.reset_problem(boost::get<std::string>(config.at("algorithm")),
                           boost::get<int>(config.at("T_x")) *
                               boost::get<int>(config.at("T_y")));
    }
}

void FiniteStateMachine::st_RESET() {
    std::cout << "FSM:: RESET state" << std::endl;
    llrs_problem_path = server_handler.get_llrs_config_file();
    llrs.setup(llrs_problem_path, false, 1);
    trigger_detector.getAWG()->stop_card();
    std::cout << "Starting AWG stream" << std::endl;
    auto awg = trigger_detector.getAWG();
    auto tb = awg->allocate_transfer_buffer(
        trigger_detector.get_samples_per_idle_segment());

    if (awg->get_idle_segment_wfm()) {
        llrs.get_idle_wfm(tb, trigger_detector.get_samples_per_idle_segment());
    } else {
        awg->fill_transfer_buffer(
            tb, trigger_detector.get_samples_per_idle_segment(), 0);
    }
    trigger_detector.setup(tb);
    trigger_detector.stream();
    server_handler.send_200();
}

void FiniteStateMachine::st_CLOSE_AWG() {
    std::cout << "FSM:: CLOSE AWG state" << std::endl;
    trigger_detector.getAWG()->stop_card();
    trigger_detector.getAWG()->close_card();
    server_handler.send_200();
}

void FiniteStateMachine::st_RESTART_AWG() {
    trigger_detector.getAWG()->configure();
    llrs.reset_awg(false, 1);
    std::cout << "Starting AWG stream" << std::endl;
    auto awg = trigger_detector.getAWG();
    auto tb = awg->allocate_transfer_buffer(
        trigger_detector.get_samples_per_idle_segment());

    if (awg->get_idle_segment_wfm()) {
        llrs.get_idle_wfm(tb, trigger_detector.get_samples_per_idle_segment());
    } else {
        awg->fill_transfer_buffer(
            tb, trigger_detector.get_samples_per_idle_segment(), 0);
    }
    trigger_detector.setup(tb);
    trigger_detector.stream();
    server_handler.send_200();
}

void FiniteStateMachine::st_CONFIG_PSF() {
    std::cout << "FSM:: CONFIG PSF state" << std::endl;
    std::string str = (PSF_TRANSLATOR_PATH) + " default";
    const char *psf_translator = str.c_str();
    const char *command = "python3 ";
    char fullCommand[256];
    snprintf(fullCommand, sizeof(fullCommand), "%s%s", command, psf_translator);
    int result = system(fullCommand);
    llrs.reset_psf(std::string("default.bin"));
    server_handler.send_200();
}
void FiniteStateMachine::st_CONFIG_WAVEFORM() {
    std::cout << "FSM:: CONFIG WAVEFORM state" << std::endl;
    llrs.reset_waveform_table();
    trigger_detector.getAWG()->stop_card();
    std::cout << "Starting AWG stream" << std::endl;
    auto awg = trigger_detector.getAWG();
    auto tb = awg->allocate_transfer_buffer(
        trigger_detector.get_samples_per_idle_segment());

    if (awg->get_idle_segment_wfm()) {
        llrs.get_idle_wfm(tb, trigger_detector.get_samples_per_idle_segment());
    } else {
        awg->fill_transfer_buffer(
            tb, trigger_detector.get_samples_per_idle_segment(), 0);
    }
    trigger_detector.setup(tb);
    trigger_detector.stream();
    server_handler.send_200();
}

void FiniteStateMachine::st_LLRS_EXEC() {
    std::cout << "FSM:: LLRS_EXEC state" << std::endl;
    assert(trigger_detector.getAWG()->get_current_step() == 1);
    std::cout << "Starting the LLRS" << std::endl;
    llrs.execute();
    llrs_metadata.push_back(llrs.getMetadata());
    std::cout << "Done LLRS::Execute" << std::endl;
    trigger_detector.reset();
}

void FiniteStateMachine::st_FAULT() {
    std::cout << "FSM:: FAULT state" << std::endl;
}

void FiniteStateMachine::st_EXIT() {
    std::cout << "FSM:: EXIT state" << std::endl;
}

void FiniteStateMachine::saveMetadata(std::string dirPath) {
    std::string parentDir = dirPath.substr(0, dirPath.find_last_of('/'));

    for (size_t i = 0; i < llrs_metadata.size(); i++) {
        nlohmann::json json_data;
        std::string filePath =
            parentDir + "/metadata_" + std::to_string(i) + ".json";

        // Save general data
        json_data["nt_x"] = llrs_metadata.at(i).getNtx();
        json_data["nt_y"] = llrs_metadata.at(i).getNty();
        json_data["cycles"] = llrs_metadata.at(i).getNumCycles();
        json_data["runtime_data"] = llrs_metadata.at(i).getRuntimeData();
        json_data["target_met"] = llrs_metadata.at(i).getTargetMet();

        // Save cycles data
        const std::vector<std::vector<Reconfig::Move>> &movesPerCycle =
            llrs_metadata.at(i).getMovesPerCycle();
        const std::vector<std::vector<int32_t>> &atomConfigs =
            llrs_metadata.at(i).getAtomConfigs();

        nlohmann::json cycles_data;
        // Save the initial atom configuration
        for (int cycle_index = 0;
             cycle_index < llrs_metadata.at(i).getNumCycles(); ++cycle_index) {
            nlohmann::json cycle_data;
            cycle_data["initial_atom_config"] = atomConfigs[cycle_index];

            nlohmann::json jsonMoves;
            if (movesPerCycle.size() > cycle_index) {
                for (const auto &move : movesPerCycle[cycle_index]) {
                    jsonMoves.push_back({std::get<0>(move), std::get<1>(move),
                                         std::get<2>(move), std::get<3>(move),
                                         std::get<4>(move)});
                }
            }
            cycle_data["moves"] = jsonMoves;

            cycles_data.push_back(cycle_data);
        }
        json_data["Cycles"] = cycles_data;

        // Output the JSON data object to the filepath
        std::ofstream outfile(filePath, std::ios::out);
        if (!outfile.is_open()) {
            std::cout << "Error opening file: " << filePath << std::endl;
            return;
        }
        outfile << json_data.dump(2);
    }
}
