/**
 * @brief   Entry point for the Low Latency FSM.
 * @date    Oct 2023
 */

#include "awg.hpp"
#include "fsm.hpp"
#include "llcs/common.hpp"
#include "server.hpp"
#include "trigger-detector.hpp"

std::atomic<bool> g_interrupted(false);

void SignalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    g_interrupted.store(true);
}

int main(int argc, char *argv[]) {

    Server *server = new Server();
    TriggerDetector<AWG> *td = new TriggerDetector<AWG>();
    FiniteStateMachine<AWG> *fsm = new FiniteStateMachine<AWG>(server, td);

    fsm->runFSM();

    delete td;
    delete server;
    delete fsm;

    std::cout << "Program terminated gracefully" << std::endl;

    return SYS_OK;
}
