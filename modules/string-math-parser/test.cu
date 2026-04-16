#include <iostream>
#include <vector>
#include <tuple>
#include <memory>
#include "string-math-parser.h"

int main() {
    // auto result = StringMathParser::evaluate("-2.323e3*323/(2+3)-4*5*+(-$x$-2)/3*-1", {{"x", 4}});
    // auto result = StringMathParser::evaluate("-100+-$x$*-(3/-$y$)", {{"x", 4}});
    auto result = StringMathParser::evaluate("2*2", {{}});

    std::cout << "result: " << result << std::endl;

    return 0;
}