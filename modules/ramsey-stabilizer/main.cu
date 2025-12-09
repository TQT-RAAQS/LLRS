#include "ramsey-stabilizer.h"

int main() {
    
    RamseyStabilizer rs("default.yml");

    rs.start();

    std::string input;
    while (true) {
        std::cout << "Enter command (quit to exit): ";
        std::getline(std::cin, input);

        if (input == "quit") {
            break;
        }

        std::cout << "You entered: " << input << std::endl;
    }

    rs.stop();

    return 0;
}