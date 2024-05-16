/**
 * @brief   Class that implements a programmable finite state machine for
 * execution of low latency procedures.
 * @date    Oct 2023
 */

#include "fsm.hpp"

using namespace std::this_thread; // sleep_for, sleep_until
using namespace std::chrono;      // nanoseconds, system_clock, seconds

/**
 * @brief Constructor of the Finite State Machine class
 *
 * Construct state objects for all states in the FSM. Connect the state
 * transitions between these states.
 */
template <typename AWG_T>
FiniteStateMachine<AWG_T>::FiniteStateMachine(Server *s,
                                              TriggerDetector<AWG_T> *td) {
    std::cout << "FSM:: constructor" << std::endl;

    // 1. Create the ST_FAULT state
    State *fault_state =
        new State([this]() { this->st_FAULT(); }, []() { return -1; }, NULL);
    states.insert({ST_FAULT, fault_state});

    // Create dynamic state action function map
    dyn_state_action_func_map = {
        {M_LLRS, [this]() { this->st_LLRS_EXEC(); }},
        {M_CLO, [this]() { this->st_CLO_EXEC(); }},
        {M_RYDBERG, [this]() { this->st_RYDBERG_EXEC(); }}};

    // Configure static states of the FSM
    setupFSM();

    // Begin the state machine the in the ST_IDLE state
    currentState = states[ST_BEGIN];

    // 3. Initiliaze the network server handle
    server = s;

    // 4. Initiliaze the trigger detector
    trigger_detector = td;
}

/**
 * @brief The destructor of the Finite State Machine class.
 *
 * For every dynamically allocated state object, free the object.
 */
template <typename AWG_T> FiniteStateMachine<AWG_T>::~FiniteStateMachine() {
    std::cout << "FSM:: destructor" << std::endl;

    states.clear();

    for (auto stateVector : programmable_states) {
        for (State *state : stateVector) {
            delete state;
        }
        stateVector.clear();
    }
    programmable_states.clear();

    if (l != nullptr) {
        delete l;
    }
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
    State *idle_state = new State([this]() { this->st_IDLE(); },
                                  [this]() { return server->listen(); }, f);
    State *llrs_state = new State([this]() { this->st_LLRS_EXEC(); },
                                  [this]() { return 1; }, f);
    State *config_hw_state =
        new State([this]() { this->st_CONFIG_HW(); }, []() { return 1; }, f);
    State *config_sm_state =
        new State([this]() { this->st_CONFIG_SM(); }, []() { return 1; }, f);
    State *ready_state =
        new State([this]() { this->st_READY(); },
                  [this]() {
                      int t;
                      while (true) {
                          t = trigger_detector->detectTrigger(6000);
                          if (t != NO_HW_TRIG) {
                              return numExperiments;
                          }
                          return -1;
                      }
                  },
                  f);
    State *trigger_done_state =
        new State([this]() { this->st_TRIGGER_DONE(); }, []() { return 1; }, f);
    State *last_trigger_done_state =
        new State([this]() { this->st_LAST_TRIGGER_DONE(); },
                  [this]() { return server->listen(); }, f);
    State *reset_state =
        new State([this]() { this->st_RESET(); }, []() { return 1; }, f);
    State *close_awg = new State([this]() { this->st_CLOSE_AWG(); },
                                 [this]() { return server->listen(); }, f);
    State *restart_awg =
        new State([this]() { this->st_RESTART_AWG(); }, []() { return 1; }, f);
    State *exit_state =
        new State([this]() { this->st_EXIT(); }, []() { return 1; }, NULL);

    states.insert({ST_BEGIN, begin});
    states.insert({ST_IDLE, idle_state});
    states.insert({ST_CONFIG_HW, config_hw_state});
    states.insert({ST_CONFIG_SM, config_sm_state});
    states.insert({ST_READY, ready_state});
    states.insert({ST_TRIGGER_DONE, trigger_done_state});
    states.insert({ST_LAST_TRIGGER_DONE, last_trigger_done_state});
    states.insert({ST_RESET, reset_state});
    states.insert({ST_CLOSE_AWG, close_awg});
    states.insert({ST_RESTART_AWG, restart_awg});
    states.insert({ST_EXIT, exit_state});

    // Configure static transitions

    begin->addStateTransition(1, idle_state);

    idle_state->addStateTransition(0, config_hw_state);
    idle_state->addStateTransition(2, ready_state);
    idle_state->addStateTransition(3, exit_state);
    idle_state->addStateTransition(4, idle_state);
    idle_state->addStateTransition(5, reset_state);
    idle_state->addStateTransition(6, close_awg);

    config_hw_state->addStateTransition(1, config_sm_state);
    config_sm_state->addStateTransition(1, idle_state);

    ready_state->addStateTransition(1, llrs_state);
    llrs_state->addStateTransition(1, last_trigger_done_state);

    last_trigger_done_state->addStateTransition(3, idle_state);

    reset_state->addStateTransition(1, idle_state);

    close_awg->addStateTransition(7, restart_awg);

    restart_awg->addStateTransition(1, idle_state);

    idle_state->setType(ST_IDLE);
    config_hw_state->setType(ST_CONFIG_HW);
    config_sm_state->setType(ST_CONFIG_SM);
    ready_state->setType(ST_READY);
    trigger_done_state->setType(ST_TRIGGER_DONE);
    last_trigger_done_state->setType(ST_LAST_TRIGGER_DONE);
    exit_state->setType(ST_EXIT);
    f->setType(ST_FAULT);
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
    State *last_trigger_done_state = states[ST_LAST_TRIGGER_DONE];
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
            case M_CLO:
                state_to_add->setType(ST_CLO_EXEC);
                break;
            case M_RYDBERG:
                state_to_add->setType(ST_RYDBERG_EXEC);
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
                    s->addStateTransition(trig_num, last_trigger_done_state);
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

    return;
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

    return;
}

void createFolder(std::string &folderPath) {
    if (!std::experimental::filesystem::is_directory(folderPath) ||
        !std::experimental::filesystem::exists(folderPath)) {
        std::experimental::filesystem::create_directory(folderPath);
    }
}

using json = nlohmann::json;
template <typename AWG_T>
void FiniteStateMachine<AWG_T>::saveMetadata(std::string dirPath) {
    json json_data;

    std::string filePath;
    std::string parentDir = dirPath.substr(0, dirPath.find_last_of('/'));
    std::string mainLLRSDir = parentDir.substr(0, parentDir.find_last_of('/'));
    createFolder(mainLLRSDir);
    createFolder(parentDir);

    for (int i = 0; i < llrs_metadata.size(); i++) {
        filePath =
            dirPath + "metadata_" + std::to_string(i) + ".json"; // FIX THIS
        // Extract Metadata and the number of trials and repetitions
        const int Nt_x = llrs_metadata.at(i).getNtx();
        const int Nt_y = llrs_metadata.at(i).getNty();
        const int numCycles = llrs_metadata.at(i).getNumCycles();
        const std::vector<std::vector<Reconfig::Move>> &movesPerCycle =
            llrs_metadata.at(i).getMovesPerCycle();
        const std::vector<std::vector<int32_t>> &atomConfigs =
            llrs_metadata.at(i).getAtomConfigs();
        const nlohmann::json &runtimeData =
            llrs_metadata.at(i).getRuntimeData();

        // Save the number of cycles executed and the dimensions of the array
        json_data["Nt_x"] = Nt_x;
        json_data["Nt_y"] = Nt_y;
        json_data["Cycles"] = numCycles;
        json_data["runtime_data"] = runtimeData;

        // Save the initial atom configuration
        for (int cycle_index = 0; cycle_index < numCycles; ++cycle_index) {
            json cycle_data;

            cycle_data["starting_atom_config"] = atomConfigs[cycle_index];

            if (movesPerCycle.size() > cycle_index) {
                nlohmann::json jsonMoves;
                for (const auto &move : movesPerCycle[cycle_index]) {
                    jsonMoves.push_back({std::get<0>(move), std::get<1>(move),
                                         std::get<2>(move), std::get<3>(move),
                                         std::get<4>(move)});
                }
                cycle_data["moves"] = jsonMoves;
            }
            if (cycle_index < numCycles - 1) {
                cycle_data["final_atom_configuration"] =
                    atomConfigs[cycle_index + 1];
            }

            json_data["Cycle " + std::to_string(cycle_index)] = cycle_data;
        }

        // Output the JSON data object to the filepath
        std::ofstream outfile(filePath, std::ios::out);
        if (!outfile.is_open()) {
            std::cout << "Error opening file: " << filePath << std::endl;
            return;
        }
        outfile << json_data.dump(2);
        outfile.close();
    }
    std::ofstream outfile(dirPath + "saving.done", std::ios::out);
    if (!outfile.is_open()) {
        std::cout << "Error opening file: " << filePath << std::endl;
        return;
    }
    outfile.close();
}

/**
 * @brief The primary execution member function of the FSM class.
 *
 * Begin with the idle state and execute each state and perform the appropriate
 * state transition.
 */
template <typename AWG_T> void FiniteStateMachine<AWG_T>::runFSM() {
    State *nextStateToRun;

    while (currentState != nullptr) {
        currentState->executeState();

        currentState = currentState->getNextState();
    }

    return;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * Static State definitions
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * */

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_BEGIN() {
    std::cout << "Starting LLRS setup" << std::endl;
    auto awg = trigger_detector->getAWG();
    l = new LLRS<AWG_T>{awg};
    l->setup("config.yml", false, 1);

    std::cout << "Starting AWG stream" << std::endl;
	auto tb = awg->allocate_transfer_buffer(
		trigger_detector->get_samples_per_idle_segment());

    if (awg->get_idle_segment_wfm()) {
       l->get_idle_wfm(tb, trigger_detector->get_samples_per_idle_segment());
    } else {
		awg->fill_transfer_buffer(tb, trigger_detector->get_samples_per_idle_segment(), 0);
    }

    trigger_detector->setup(tb);
    trigger_detector->stream();
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_IDLE() {
    std::cout << "FSM:: IDLE state" << std::endl;
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_CONFIG_HW() {
    std::cout << "FSM:: CONFIG_HW state" << std::endl;

    // receive hdf5 filepath from the workstation
    // std::string filepath = server->get_config_file_path();

    // filepath is now the yaml config file
    // filepath = H5Wrapper::convertHWConfig(filepath);

    // configure the AWG
    // auto awg = trigger_detector->getAWG();
    // awg->read_config(filepath);

    // l->small_setup(filepath);

    HWconfigured = true;
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_CONFIG_SM() {
    std::cout << "FSM:: CONFIG_SM state" << std::endl;

    // If the FSM loop has already been run before then reset transition map.

    // if (numExperiments > 1) {
    //     resetTransitions();
    // }
    // std::string filepath = server->get_config_file_path();

    // programStates(filepath);
    SMconfigured = true;
    numExperiments = 1;
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_READY() {
    std::cout << "FSM:: READY state" << std::endl;

    std::cout << "Awaiting Hardware Trigger..." << std::endl;
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_TRIGGER_DONE() {
    std::cout << "FSM:: TRIGGER_DONE state" << std::endl;
    numExperiments++;
}

template <typename AWG_T>
void FiniteStateMachine<AWG_T>::st_LAST_TRIGGER_DONE() {
    std::cout << "FSM:: LAST_TRIGGER_DONE state" << std::endl;

    HWconfigured = false;
    SMconfigured = false;

    saveMetadata(server->get_metadata_file_path());
    llrs_metadata.clear();
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_RESET() {

    if (l != nullptr) {
        delete l;
    }

    auto awg = trigger_detector->getAWG();
    l = new LLRS<AWG_T>{awg};
    l->setup("config.yml", false, 1);

    std::cout << "Starting AWG stream" << std::endl;
   	auto tb = awg->allocate_transfer_buffer(
		trigger_detector->get_samples_per_idle_segment());

    if (awg->get_idle_segment_wfm()) {
       l->get_idle_wfm(tb, trigger_detector->get_samples_per_idle_segment());
    } else {
		awg->fill_transfer_buffer(tb, trigger_detector->get_samples_per_idle_segment(), 0);
    }
    trigger_detector->setup(tb);
    trigger_detector->stream();

}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_CLOSE_AWG() {
    std::cout << "FSM:: CLOSE AWG state" << std::endl;
    trigger_detector->getAWG()->stop_card();
    trigger_detector->getAWG()->close_card();
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_RESTART_AWG() {

    std::cout << "FSM:: CLOSE AWG state" << std::endl;

    auto awg = trigger_detector->getAWG();
    const int64_t samples_per_segment = awg->get_samples_per_segment();

    std::cout << "Starting AWG stream" << std::endl;
   	auto tb = awg->allocate_transfer_buffer(
		trigger_detector->get_samples_per_idle_segment());

    if (awg->get_idle_segment_wfm()) {
       l->get_idle_wfm(tb, trigger_detector->get_samples_per_idle_segment());
    } else {
		awg->fill_transfer_buffer(tb, trigger_detector->get_samples_per_idle_segment(), 0);
    }
    trigger_detector->setup(tb);
    trigger_detector->stream();
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_FAULT() {
    std::cout << "FSM:: FAULT state" << std::endl;

    return;
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_EXIT() {

    std::cout << "FSM:: EXIT state" << std::endl;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * Dynamic/Programmable State definitions
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * */

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_LLRS_EXEC() {
    std::cout << "      FSM:: LLRS_EXEC state" << std::endl;

    std::shared_ptr<AWG_T> awg{trigger_detector->getAWG()};

    // LLRS<AWG_T> *l = new LLRS<AWG_T>{awg};

    std::cout << "      FSM:: LLRS_EXEC state" << std::endl;
    int current_step = awg->get_current_step();
    std::cout << "current step: " << current_step << std::endl;

    assert(current_step == 1);
    std::cout << "Starting the LLRS" << std::endl;
    l->execute();
    llrs_metadata.emplace_back(l->getMetadata());
    std::cout << "Done LLRS::Execute" << std::endl;

    // should be on llrs idle segment 4
    // td->resetDetectionSegments();
    assert(current_step == 1);
    awg->seqmem_update(1, 0, 1, 0,
                       SPCSEQ_ENDLOOPALWAYS); // this is slow
    trigger_detector->busyWait();

    current_step = awg->get_current_step();
    std::cout << "Current step is: " << current_step << std::endl;
    assert(current_step == 0);

    // Ensure LLRS Idle is pointing to itself // move this into LLRS reset
    awg->seqmem_update(1, 0, 1, 1, SPCSEQ_ENDLOOPALWAYS);
    trigger_detector->busyWait();

#ifdef LOGGING_RUNTIME
    l->clean();
#endif
    return;
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_CLO_EXEC() {
    std::cout << "      FSM:: CLO_EXEC state" << std::endl;

    return;
}

template <typename AWG_T> void FiniteStateMachine<AWG_T>::st_RYDBERG_EXEC() {
    std::cout << "      FSM:: RYDBERG_EXEC state" << std::endl;

    return;
}

template class FiniteStateMachine<AWG>;
