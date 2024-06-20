/**
 * @brief   Class that implements a state in the finite state machine.
 * @date    Oct 2023
 */

#include "state.hpp"

/**
 * @brief Constructor of the State class
 *
 */
State::State(std::function<void()> ac_func, std::function<int()> tr_func,
             State *fault_transition) {
    std::cout << "STATE:: constructor" << std::endl;

    action_function = ac_func;
    transition_function = tr_func;

    transitions.push_back(std::make_tuple(-1, fault_transition));
}

/**
 * @brief The destructor of the State class.
 *
 * For every dynamically allocated state object, free the object.
 */
State::~State() {
    std::cout << "STATE:: destructor " << this->getType() << std::endl;

    this->resetTransitionMap();
}

void State::resetTransitionMap() {

    transitions.erase(std::remove_if(transitions.begin(), transitions.end(),
                                     [](const std::tuple<int, State *> &t) {
                                         return std::get<0>(t) != -1;
                                     }),
                      transitions.end());

    return;
}

void State::addStateTransition(int data, State *state_transition) {

    transitions.insert(transitions.end() - 1,
                       std::make_tuple(data, state_transition));

    return;
}

void State::executeState() {

    action_function();

    return;
}

State *State::getNextState() {

    int result = transition_function();
    int transition_data;
    State *next_state = std::get<1>(transitions.back());
    for (auto &transition : transitions) {
        transition_data = std::get<0>(transition);
        if (result == transition_data) {
            next_state = std::get<1>(transition);
            break;
        }
    }
    return next_state;
}

State *State::getNextProgrammableState(int result) {
    int transition_data;
    State *next_state;

    for (auto &transition : transitions) {
        transition_data = std::get<0>(transition);

        if (result == transition_data) {
            next_state = std::get<1>(transition);
            return next_state;
        }
    }

    // No transition picked, return fault transtions, in principle, this
    // should'nt happen becase we should be returning a faulty value if the
    // transition fails
    State *fault_state = std::get<1>(transitions.back());
    return fault_state;
}

void State::printState() {

    std::cout << "State " << this << std::endl;
    std::cout << "State Type -> " << this->getType() << std::endl;
    std::cout << "Transition Map        => " << std::endl;

    for (auto &transition : transitions) {
        std::cout << "      Transition Data  -> " << std::get<0>(transition)
                  << std::endl;
        std::cout << "      State Transition -> " << (std::get<1>(transition))
                  << std::endl;

        if (std::get<1>(transition) != nullptr) {
            std::cout << "      Next State Type  -> "
                      << (std::get<1>(transition))->getType() << std::endl;
        }

        std::cout << std::endl;
    }

    return;
}

void State::setType(StateType tp) { type = tp; }

std::string State::getType() {
    switch (type) {
    case ST_FAULT:
        return "ST_FAULT";
    case ST_EXIT:
        return "ST_EXIT";
    case ST_IDLE:
        return "ST_IDLE";
    case ST_READY:
        return "ST_READY";
    case ST_TRIGGER_DONE:
        return "ST_TRIGGER_DONE";
    case ST_LLRS_EXEC:
        return "ST_LLRS_EXEC";
    case ST_PROCESS_SHOT:
        return "ST_PROCESS_SHOT";
    case ST_CLOSE_AWG:
        return "ST_CLOSE_AWG";
    case ST_RESTART_AWG:   
        return "ST_RESTART_AWG";
    case ST_BEGIN:
        return "ST_BEGIN";
    case ST_RESET:
        return "ST_RESET";
    case ST_CONFIG_PSF:
        return "ST_CONFIG_PSF";
    case ST_CONFIG_WAVEFORM:
        return "ST_CONFIG_WAVEFORM";
    default:
        return "UNKNOWN_STATE";
    }
}

int State::getTransitionSize() { return transitions.size(); }