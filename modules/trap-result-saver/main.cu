#include "trap-result-saver.h"
#include <iostream>

int main() {
    std::string config_name = "default.yml";
    TrapResultSaver result_saver(config_name);

    result_saver.start();

    std::string input;
    while (true) {
        std::cout << "Enter command (quit to exit): ";
        std::getline(std::cin, input);

        if (input == "quit") {
            break;
        }

        std::cout << "You entered: " << input << std::endl;
    }

    result_saver.stop();

    return 0;
}
