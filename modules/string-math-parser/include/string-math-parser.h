#ifndef _STRING_MATH_PARSER_H
#define _STRING_MATH_PARSER_H

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>

namespace StringMathParser {

    class TermNode : public std::enable_shared_from_this<TermNode> {
        public:
            const std::string term;

            bool can_be_evaluated = false;
            bool evaluated = false;
            double value = 0.0;
            char operator_char = ';';
            std::shared_ptr<TermNode> left = nullptr;
            std::shared_ptr<TermNode> right = nullptr;
            std::shared_ptr<TermNode> first_child = nullptr;
            std::shared_ptr<TermNode> parent = nullptr;

            TermNode(const std::string& term) : term(term) {}

            bool decompose();
    };

    double evaluate(const std::string& expression, const std::unordered_map<std::string, double>& variables);

}

#endif