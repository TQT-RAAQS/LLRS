#include "string-math-parser.h"
#include <memory>

bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

bool is_operator(char c) {
    return c == '+' || c == '-' || c == '*' || c == '/';
}

std::string remove_whitespace(const std::string& str) {
    std::string result;
    for (char c : str) {
        if (!std::isspace(c)) {
            result += c;
        }
    }
    return result;
}

namespace {

size_t parse_variable_end(const std::string& expression, size_t start) {
    size_t pos = start + 1;
    while (pos < expression.size() && expression[pos] != '$') {
        ++pos;
    }
    if (pos >= expression.size() || expression[pos] != '$' || pos == start + 1) {
        throw std::runtime_error("Mismatched variable delimiters in expression.");
    }
    return pos + 1;
}

size_t parse_parenthesized_term_end(const std::string& expression, size_t start) {
    size_t pos = start + 1;
    int parentheses_count = 1;
    while (pos < expression.size() && parentheses_count > 0) {
        if (expression[pos] == '(') {
            ++parentheses_count;
        } else if (expression[pos] == ')') {
            --parentheses_count;
        }
        ++pos;
    }
    if (parentheses_count != 0) {
        throw std::runtime_error("Mismatched parentheses in expression.");
    }
    return pos;
}

size_t parse_number_end(const std::string& expression, size_t start) {
    size_t pos = start;
    size_t position_e = 0; // 0 for before E, 1 immediately after E, 2 for after E
    bool decimal_flag = false;
    bool sign_flag;

    if (expression.at(start) == '+' || expression.at(start) == '-') {
        sign_flag = true;
        pos = start + 1;
    }
    else {
        sign_flag = false;
    }

    while (pos < expression.size()) {
        if (position_e == 0) {
            if (is_digit(expression.at(pos))) {
                ++pos;
            } else if (expression.at(pos) == 'E' || expression.at(pos) == 'e') {
                position_e = 1;
                ++pos;
            } else if (expression.at(pos) == '.') {
                if (decimal_flag) {
                    throw std::runtime_error("Invalid format for number with multiple decimal points.");
                }
                decimal_flag = true;
                ++pos;
            } else {
                break;
            }
        } else if (position_e == 1) {
            if (expression.at(pos) == '+' || expression.at(pos) == '-' || is_digit(expression.at(pos))) {
                position_e = 2;
                ++pos;
            } else {
                throw std::runtime_error("Invalid format for scientific notation in number.");
            }
        } else { // position_e == 2
            if (is_digit(expression.at(pos))) {
                ++pos;
            } else {
                return pos;
            }
        }
    }
    if (sign_flag  && pos == start + 1) {
        throw std::runtime_error("Invalid format for number with only a sign and no digits.");
    }
    return pos;
}

} // namespace

size_t get_term_end(const std::string& expression, size_t start) {
    size_t pos = start;
    
    // Variables
    if (expression.at(pos) == '$') {
        return parse_variable_end(expression, start);
    }

    // Term enclosed in paranthesis
    if (expression.at(pos) == '(') {
        return parse_parenthesized_term_end(expression, start);
    }
    
    // Number
    if (is_digit(expression.at(pos))) {
        return parse_number_end(expression, start);
    }
    
    return pos;
}

std::tuple<std::vector<std::string>, std::vector<char>> parse_term(const std::string& expression) {
    const auto end = expression.size();
    auto counter = size_t{0};
    std::vector<std::string> terms;
    std::vector<char> operators;
    while (counter < end) {
        // Parsing a term
        auto term_end = get_term_end(expression, counter);
        auto term = expression.substr(counter, term_end - counter);
        terms.push_back(term);
        if (term_end == counter) {
            throw std::runtime_error("Invalid term in expression.");
        }
        if (term_end == end) {
            break;
        }
        if (term_end > end) {
            throw std::runtime_error("Term extends beyond the end of the expression.");
        }

        // Parsing an operator
        counter = term_end;
        if (!is_operator(expression.at(term_end))) {
            throw std::runtime_error("Expected operator after term in expression.");
        }
        auto term_operator = expression.at(term_end);
        operators.push_back(term_operator);
        counter = term_end + 1;
    }

    return {terms, operators};
}

bool StringMathParser::TermNode::decompose() {
    auto output = parse_term(this->term);
    auto terms = std::get<0>(output);
    auto operators = std::get<1>(output);

    if (terms.size() == 0) {
        throw std::runtime_error("No terms found in expression.");
    }

    if (operators.size() != terms.size() - 1) {
        throw std::runtime_error("Number of operators does not match number of terms in expression.");
    }

    bool expandable_single_term = terms.size() == 1 && terms.at(0).at(0) == '(' && terms.at(0).back() == ')';
    if (terms.size() > 1 || expandable_single_term) {
        // auto pointer = std::shared_ptr<TermNode>(terms.back());
        // for (size_t i = terms.size(); i > 0; --i) {
            
        // }

        return false;
    }
}

double StringMathParser::evaluate(const std::string& input_expression, const std::unordered_map<std::string, double>& variables) {
    auto clean_expression = remove_whitespace(input_expression);

    auto root = std::make_shared<TermNode>(clean_expression);
    auto pointer = root;

    while (!root->evaluated) {
        auto result = parse_term(pointer->term);
        auto terms = std::get<0>(result);
        auto operators = std::get<1>(result);  
        for (const auto& term : terms) {
            std::cout << "Term: " << term << std::endl;
        }
        break;
    }

    return 0.0;
}