#include <iostream>
#include <vector>
#include <tuple>
#include <memory>
#include "string-math-parser.h"

int main() {
    auto result = StringMathParser::evaluate("2.323e3*323/(2+3)", {{"x", 4}});

    return 0;
}