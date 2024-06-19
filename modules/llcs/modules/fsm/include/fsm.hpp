#ifndef FSM_HPP_
#define FSM_HPP_

#include "h5-wrapper.hpp"
#include "llcs/common.hpp"
#include "llrs.h"
#include "server.hpp"
#include "state.hpp"
#include "trigger-detector.hpp"
#include <chrono>
#include <experimental/filesystem>
#include <memory>
#include <thread>

template <typename AWG_T> class FiniteStateMachine {
    State *currentState;
    std::unordered_map<StateType, State *> states;
    std::vector<std::vector<State *>> programmable_states;
    std::unordered_map<ModuleType, std::function<void()>>
        dyn_state_action_func_map;
    
    Server server;
    TriggerDetector<AWG_T> trigger_detector;
    LLRS<AWG_T> l;
  
    int numExperiments = 0;
    std::string llrs_problem_path = "21-problem.yml";
    std::vector<typename LLRS<AWG_T>::Metadata> llrs_metadata;
    /**
     * @brief Creates all state objects for the static states of the FSM
     */
    void setupFSM();

    /**
     * @brief Reset the configurable sequence of experimental states after each
     * shot
     */
    void resetTransitions();

    /**
     * @brief Entry state of the FSM. Creates the LLRS object and streams static
     * waveforms
     */
    void st_BEGIN();

    /**
     * @brief Idle state. Waits for requests from the server to transition to
     * another state
     */
    void st_IDLE();

    /**
     * @brief Configures relevant hardware with data sent by the workstation
     */
    void st_CONFIG_HW();

    /**
     * @brief Programs the shot sequence with data sent by the workstation
     */
    void st_CONFIG_SM();

    /**
     * @brief Waits for a hardware trigger to begin commencing the experimental
     * shot sequence
     */
    void st_READY();

    /**
     * @brief Intermediary state between each hardware trigger in the experiment
     */
    void st_TRIGGER_DONE();

    /**
     * @brief State at the end of the experimental shot. Sends relevant metadata
     * to the workstation.
     */
    void st_LAST_TRIGGER_DONE();

    /**
     * @brief dumps the LLRS object and creates a new one with a different setup
     */
    void st_RESET();

    /**
     * @brief Closes AWG card for another device to take over
     */
    void st_CLOSE_AWG();

    /**
     * @brief Reopens AWG card and streams static waveforms
     */
    void st_RESTART_AWG();

    /**
     * @brief Error state
     */
    void st_FAULT();

    /**
     * @brief State to exit the FSM gracefully
     */
    void st_EXIT();

    /**
     * @brief State to execute the LLRS
     */
    void st_LLRS_EXEC();

    /**
     * @brief State to execute the CLO
     */
    void st_CLO_EXEC();

  public:
    FiniteStateMachine();
    ~FiniteStateMachine();

    /**
     * @brief The primary execution member function of the FSM class.
     *
     * Begin with the BEGIN state and execute each state and perform the
     * appropriate state transition.
     */
    void runFSM();

    /**
     * @brief Program the configurable part of the state machine, configured by
     * the workstation via a ZMQ server
     */
    void programStates(std::string filepath);

    /**
     * @brief Print all states
     */
    void printStates();

    /**
     * @brief Saves metadata of each shot to be returned to the workstation
     */
    void saveMetadata(std::string filepath);

};
#endif
