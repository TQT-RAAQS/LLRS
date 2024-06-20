#include "fsm.hpp"
using json = nlohmann::json;

/**
 * @brief Constructor of the Finite State Machine class
 *
 * Construct state objects for all states in the FSM. Connect the state
 * transitions between these states.
 */
template <typename AWG_T>
FiniteStateMachine<AWG_T>::FiniteStateMachine(): server_handler{}, trigger_detector{}, llrs{trigger_detector.getAWG()} {

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
template <typename AWG_T> FiniteStateMachine<AWG_T>::~FiniteStateMachine() {
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
template <typename AWG_T> void FiniteStateMachine<AWG_T>::setupFSM() {

    State *f = states[ST_FAULT];

    State *begin =
        new State([this]() { this->st_BEGIN(); }, []() { return 1; }, f);
    State *idle_state =
        new State([this]() { this->st_IDLE(); },
                  [this]() { return server_handler.get_request(); }, f);
   State *psf_reset_state =
        new State([this]() { this->st_CONFIG_PSF(); }, []() { return 1; }, f);
    State *waveform_reset_state =
        new State([this]() { this->st_CONFIG_WAVEFORM(); }, []() { return 1; }, f);
    State *llrs_reset_state =
        new State([this]() { this->st_RESET(); }, []() { return 1; }, f);
    State *close_awg_state =
        new State([this]() { this->st_CLOSE_AWG(); },
                  [this]() { return server_handler.get_request(); }, f);
    State *restart_awg_state =
        new State([this]() { this->st_RESTART_AWG(); }, []() { return 1; }, f);
    State *process_shot_state =
        new State([this]() { this->st_PROCESS_SHOT(); }, []() { return 1; }, f);
    State *ready_state =
        new State([this]() { this->st_READY(); },
                  [this]() {
                    return trigger_detector.detectTrigger(6000) != NO_HW_TRIG?1:-1;
                  },
                  f);
    State *llrs_state = new State([this]() { this->st_LLRS_EXEC(); },
                                  [this]() { return 1; }, f);
    State *trigger_done_state =
        new State([this]() { this->st_TRIGGER_DONE(); }, [this]() { return this->numExperiments?2:1; }, f);
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

template <typename AWG_T> void FiniteStateMachine<AWG_T>::resetTransitions() {

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

template <typename AWG_T> void FiniteStateMachine<AWG_T>::printStates() {
    for (const auto &pair : states) {
        State *state = pair.second;
        state->printState();
        std::cout << std::endl;
    }
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::runFSM() {
    server_handler.start_listening();
    while (currentState != nullptr) {
        currentState->executeState();
        currentState = currentState->getNextState();
    }
}

/* * * * * * * * Static State definitions * * * * * * * */

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_BEGIN() {
    std::cout << "FSM:: BEGIN state" << std::endl;
    llrs.setup(llrs_problem_path, false, 1); // Default problem, operator can reset the problem in the idle step
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_IDLE() {
    std::cout << "FSM:: IDLE state" << std::endl;
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_PROCESS_SHOT() {
    std::cout << "FSM:: PROCESS_SHOT state" << std::endl;

    // receive hdf5 filepath from the workstation
    std::string filepath = server_handler.get_hdf5_file_path();

    //filepath is now the yaml config file
    filepath = H5Wrapper::convertHWConfig(filepath);
    llrs_metadata.reserve(1);  // placeholder

    
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

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_READY() {
    std::cout << "FSM:: READY state" << std::endl;
    std::cout << "Awaiting Hardware Trigger..." << std::endl;
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_TRIGGER_DONE() {
    std::cout << "FSM:: TRIGGER_DONE state" << std::endl;
    server_handler.wait_for_done();
    ++numExperiments;
    if (numExperiments == 0) {
        saveMetadata(server_handler.get_hdf5_file_path());
        llrs_metadata.clear();
    }
    server_handler.send_done();
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_RESET() {
    llrs_problem_path = server_handler.get_llrs_config_file();
    llrs.setup(llrs_problem_path, false, 1);
    server_handler.send_200();
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_CLOSE_AWG() {
    std::cout << "FSM:: CLOSE AWG state" << std::endl;
    trigger_detector.getAWG()->stop_card();
    trigger_detector.getAWG()->close_card();
    server_handler.send_200();
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_RESTART_AWG() {
    trigger_detector.getAWG()->configure();
    llrs.reset_awg(false, 1);
    server_handler.send_200();
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_CONFIG_PSF() {
    std::cout << "FSM:: CONFIG PSF state" << std::endl;
    // PSF Translator
    llrs.reset_psf(std::string("psf_file"));
    server_handler.send_200();
}
template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_CONFIG_WAVEFORM() {
    std::cout << "FSM:: CONFIG WAVEFORM state" << std::endl;
    llrs.reset_waveform_table();
    server_handler.send_200();
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_LLRS_EXEC() {
    std::cout << "FSM:: LLRS_EXEC state" << std::endl;
    assert(trigger_detector.getAWG()->get_current_step() == 1);
    std::cout << "Starting the LLRS" << std::endl;
    llrs.execute();
    llrs_metadata.push_back(llrs.getMetadata());
    std::cout << "Done LLRS::Execute" << std::endl;
    trigger_detector.reset();
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_FAULT() {
    std::cout << "FSM:: FAULT state" << std::endl;
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_EXIT() {
    std::cout << "FSM:: EXIT state" << std::endl;
}

template <typename AWG_T>
void FiniteStateMachine<AWG_T>::saveMetadata(std::string dirPath) {
    std::string parentDir = dirPath.substr(0, dirPath.find_last_of('/'));

    for (size_t i = 0; i < llrs_metadata.size(); i++) {
        nlohmann::json json_data;
        std::string filePath = parentDir + "metadata_" + std::to_string(i) + ".json";

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
        for (int cycle_index = 0; cycle_index < llrs_metadata.at(i).getNumCycles(); ++cycle_index) {
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

/**
 * @brief Program the configurable part of the state machine.
 *
 * Create a state object for every action function that should run during an
 * experimental shot. Adds this state object to the list of state objects in the
 * fsm.
 *
 */
template <typename AWG_T>
void FiniteStateMachine<AWG_T>::programStates(std::string filepath) {

    State *fault_transition = states[ST_FAULT];
    State *ready_state = states[ST_READY];
    State *trigger_done_state = states[ST_TRIGGER_DONE];
    State *idle_state = states[ST_IDLE];

    fault_transition->setType(ST_FAULT);
    ready_state->setType(ST_READY);
    trigger_done_state->setType(ST_TRIGGER_DONE);

    // Read a config file and configure the transition
    std::map<std::string, std::vector<ModuleType>> data =
        H5Wrapper::parseSMConfig(filepath);

    size_t num_keys = data.size();
    std::cout << "dict size: " << num_keys << std::endl;

    State *state_to_add;
    std::vector<State *> temp;

    std::string str_trig_num;
    int trig_num;

    // // Eventually nuke this - just a config file print
    for (const auto &pair : data) {
        std::cout << pair.first << ": ";
        for (const auto &val : pair.second) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    for (const auto &pair : data) {

        std::string trig_name = pair.first;
        std::vector<ModuleType> execution_sequence = pair.second;

        str_trig_num = trig_name.substr(2);
        trig_num = std::stoi(str_trig_num); // Assume this doesn't error, can
                                            // add error checking later

        // Push new states in order into temporary list
        for (ModuleType val : execution_sequence) {
            std::function<void()> action_function =
                dyn_state_action_func_map[val];

            state_to_add = new State(
                action_function, [trig_num]() { return trig_num; },
                fault_transition); // NOTE: transition may be removed,
                                   // superfluous for now
            switch (val) {
            case M_LLRS:
                state_to_add->setType(ST_LLRS_EXEC);
                break;
            }

            temp.push_back(state_to_add);
        }

        // Connect the states in temp as a sequence
        for (size_t i = 0; i < temp.size(); i++) {
            State *s = temp[i];
            if (i + 1 < temp.size()) {
                s->addStateTransition(trig_num, temp[i + 1]);
            } else {
                std::cout << "adding transition from " << s->getType();
                // transition to the ready state to await hardware trigger if
                // this is not the last sequence
                if ((size_t)trig_num != num_keys) {

                    // transition to the done shot state at the end of every
                    // sequence
                    s->addStateTransition(trig_num, trigger_done_state);
                    std::cout
                        << " to "
                        << s->getNextProgrammableState(trig_num)->getType();
                    // prevents the transition vector for done from getting
                    // infinitely large. Used <= since there will always be a
                    // fault transition
                    if (s->getNextProgrammableState(trig_num)
                            ->getTransitionSize() <= trig_num) {
                        s->getNextProgrammableState(trig_num)
                            ->addStateTransition(trig_num, ready_state);
                        std::cout << " to "
                                  << s->getNextProgrammableState(trig_num)
                                         ->getNextProgrammableState(trig_num)
                                         ->getType();
                    }
                    std::cout << std::endl;
                }
                // transition to the idle state after the last sequence
                else {
                    // transition to the done shot 2 state at the end of the
                    // experiment
                    s->addStateTransition(trig_num, trigger_done_state);
                    std::cout
                        << " to "
                        << s->getNextProgrammableState(trig_num)->getType();
                    // prevents the transition map for done 2 from getting
                    // infinitely large.
                    if (s->getNextProgrammableState(trig_num)
                            ->getTransitionSize() == 1) {
                        s->getNextProgrammableState(trig_num)
                            ->addStateTransition(1, idle_state);
                        std::cout << " to "
                                  << s->getNextProgrammableState(trig_num)
                                         ->getNextProgrammableState(1)
                                         ->getType();
                    }
                    std::cout << std::endl;
                }
            }
        }

        // add the state transition from the ST_READY state to the new sequence
        // of states
        State *first_in_sequence = temp[0];

        ready_state->addStateTransition(trig_num, first_in_sequence);

        programmable_states.push_back(temp);
        temp.clear();
    }
}

template class FiniteStateMachine<AWG>;
