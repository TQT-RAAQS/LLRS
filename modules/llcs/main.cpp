#include "awg.hpp"
#include "fsm.hpp"
#include "llcs/common.hpp"
#include "server.hpp"
#include "trigger-detector.hpp"
#include <memory>
#include <signal.h>

void my_handler(int s) {
    printf("Caught signal %d\n", s);
    exit(1);
}

int main(int argc, char *argv[]) {
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    FiniteStateMachine<AWG> fsm{};

    fsm.runFSM();
}
