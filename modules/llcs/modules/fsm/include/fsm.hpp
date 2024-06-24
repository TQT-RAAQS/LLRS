#ifndef FSM_HPP_
#define FSM_HPP_

#include "handler.hpp"
#include "llcs-config.h"
#include "llcs/common.hpp"
#include "llrs.h"
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

    Handler server_handler;
    TriggerDetector<AWG_T> trigger_detector;
    LLRS<AWG_T> llrs;

    std::string llrs_problem_path = "21-problem.yml";
    std::vector<typename LLRS<AWG_T>::Metadata> llrs_metadata;
    std::vector<LLCSCommand> commands;
    int commands_itr = 0;
    void setupFSM();
    void resetTransitions();

    /**
     * @brief Entry state of the FSM. Setups the LLRS object based on default
     * config.
     */
    void st_BEGIN();

    /**
     * @brief Idle state. Waits for requests from the server to transition to
     * another state.
     */
    void st_IDLE();

    /**
     * @brief Resets the llrs object with a different setup
     */
    void st_RESET();

    /**
     * @brief Configures the PSF
     */
    void st_CONFIG_PSF();

    /**
     * @brief Configures the waveform
     */
    void st_CONFIG_WAVEFORM();

    /**
     * @brief Processes the shots that are read from the HDF5 file
     */
    void st_PROCESS_SHOT();

    /**
     * @brief Waits for a hardware trigger to begin commencing the experimental
     * shot sequence.
     */
    void st_READY();

    /**
     * @brief Intermediary state between each hardware trigger in the experiment
     */
    void st_TRIGGER_DONE();

    /**
     * @brief Closes AWG card for another device to take over
     */
    void st_CLOSE_AWG();

    /**
     * @brief Reopens AWG card
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
     * @brief Saves metadata of each shot to be returned to the workstation
     */
    void saveMetadata(std::string filepath);

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
     * @brief Print all states
     */
    void printStates();
};
#endif
