#include "string-math-parser.h"
#include <cctype>

namespace {

bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

bool is_operator(char c) {
    return c == '+' || c == '-' || c == '*' || c == '/';
}

std::string cleanup_expression(const std::string& str) {
    std::string result;
    for (size_t i = 0; i < str.size(); ++i) {
        char c = str[i];
        if (!std::isspace(static_cast<unsigned char>(c))) {
            result += c;
        }
    }
    return result;
}

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
    bool sign_flag = false;

    if (expression.at(start) == '+' || expression.at(start) == '-') {
        sign_flag = true;
        pos = start + 1;
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
    if (sign_flag && pos == start + 1) {
        throw std::runtime_error("Invalid format for number with only a sign and no digits.");
    }
    return pos;
}

} // namespace

size_t get_term_end(const std::string& expression, size_t start) {
    size_t pos = start;

    if (expression.at(pos) == '+' || expression.at(pos) == '-') {
        ++pos;
    }

    if (expression.at(pos) == '$') {
        return parse_variable_end(expression, pos);
    }

    if (expression.at(pos) == '(') {
        return parse_parenthesized_term_end(expression, pos);
    }

    if (is_digit(expression.at(pos))) {
        return parse_number_end(expression, pos);
    }

    return pos;
}

std::tuple<std::vector<std::string>, std::vector<char>> parse_term(const std::string& expression) {
    const auto end = expression.size();
    auto counter = size_t{0};
    std::vector<std::string> terms;
    std::vector<char> operators;

    while (counter < end) {
        auto term_end = get_term_end(expression, counter);
        if (term_end == counter) {
            throw std::runtime_error("Invalid term in expression.");
        }
        if ((expression.at(counter) == '+' || expression.at(counter) == '-')) {
            if (term_end == counter + 1) {
                throw std::runtime_error("Invalid term in expression with only an operator and no operand.");
            }
            if ((term_end - counter > 2 || expression.at(counter + 1) != '1')) {
                terms.emplace_back(std::string(1, expression.at(counter)) + "1");
                if (operators.size() == 0) {
                    operators.push_back('*');
                } else {
                    if (operators.back() == '/') {
                        operators.push_back('/');
                    } else {
                        operators.push_back('*');
                    }
                }
                counter += 1;
            }
        }
        
        terms.emplace_back(expression.substr(counter, term_end - counter));
        if (term_end == end) {
            break;
        }
        if (term_end > end) {
            throw std::runtime_error("Term extends beyond the end of the expression.");
        }

        counter = term_end;
        if (!is_operator(expression.at(term_end))) {
            throw std::runtime_error("Expected operator after term in expression.");
        }
        operators.push_back(expression.at(term_end));
        counter = term_end + 1;
    }

    return {terms, operators};
}

void print_tree_dfs(const std::shared_ptr<StringMathParser::TermNode>& node, int depth) {
    if (!node) {
        return;
    }

    std::string indent(static_cast<size_t>(depth) * 2, ' ');
    std::cout << indent << "Term: " << node->term
              << " | Operator: '" << (node->operator_char ? node->operator_char : '-')
              << "' | Can evaluate: " << node->can_be_evaluated << std::endl;

    if (node->first_child) {
        print_tree_dfs(node->first_child, depth + 1);
    }
    if (node->right) {
        print_tree_dfs(node->right, depth);
    }
}

std::shared_ptr<StringMathParser::TermNode> create_expression_tree(const std::string& expression) {
    auto root = std::make_shared<StringMathParser::TermNode>(expression);

    auto current = root;
    while (true) {
        const bool was_decomposed = current->can_be_evaluated ? true : current->decompose();
        if (!was_decomposed) {
            current = current->first_child;
            continue;
        }

        if (current->right) {
            current = current->right;
            continue;
        }

        if (current->parent) {
            current = current->parent;
            current->can_be_evaluated = true;
        } else {
            break;
        }
    }

    return root;
}

double evaluate_node(const std::shared_ptr<StringMathParser::TermNode>& node, const std::unordered_map<std::string, double>& variables) {
    if (node->first_child) {

        bool value_reset = true;
        double value;
        auto& child = node->first_child;

        std::vector<double> child_values;
        std::vector<char> child_operators;
        while (child) {
            if (value_reset) {
                value = child->value;
                value_reset = false;
            }
            switch (child->operator_char) {
                case '+':
                    child_values.push_back(value);
                    child_operators.push_back('+');
                    value = 1.0;
                    break;
                case '-':
                    child_values.push_back(value);
                    child_operators.push_back('-');
                    value = 1.0;
                    break;
                case '*':
                    value *= child->value;
                    break;
                case '/':
                    value /= child->value;
                    break;
                case ';':
                    child_values.push_back(value);
                    break;
            }
        }

    } else {
        if (node->term.front() == '$') {
            return variables.at(node->term.substr(1, node->term.size() - 2));
        }
        return std::stod(node->term);
    }
}

double evaluate_tree(std::shared_ptr<StringMathParser::TermNode> root, const std::unordered_map<std::string, double>& variables) {
    auto pointer = root;
    while (true) {
        // Go as far right as possible
        if (pointer->right && !pointer->right->evaluated) {
            pointer = pointer->right;
            continue;
        }

        // Go to the first child if it exists and is not evaluated
        if (pointer->first_child && !pointer->first_child->evaluated) {
            pointer = pointer->first_child;
            continue;
        }
        
        // Evaluate the current node
        pointer->value = evaluate_node(pointer, variables);
        pointer->evaluated = true;
        
        // After evaluation is done, go left or up if possible, otherwise break
        if (pointer->left) {
            pointer = pointer->left;
        } else if (pointer->parent) {
            pointer = pointer->parent;
        } else {
            break;
        }
    }

    return root->value;
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

    const bool expandable_single_term = terms.size() == 1 && terms.at(0).front() == '(' && terms.at(0).back() == ')';
    if (terms.size() > 1) {
        auto pointer = std::make_shared<TermNode>(terms.back());
        for (size_t i = terms.size() - 1; i > 0; --i) {
            pointer->parent = this->shared_from_this();

            auto new_node = std::make_shared<TermNode>(terms.at(i - 1));
            new_node->operator_char = operators.at(i - 1);
            new_node->right = pointer;
            pointer->left = new_node;

            pointer = new_node;
        }
        this->first_child = pointer;

        return false;
    }
    if (expandable_single_term) {
        auto new_term = terms.at(0).substr(1, terms.at(0).size() - 2);
        auto pointer = std::make_shared<TermNode>(new_term);
        pointer->parent = this->shared_from_this();
        this->first_child = pointer;

        return false;
    }

    this->can_be_evaluated = true;
    return true;
}

double StringMathParser::evaluate(const std::string& input_expression, const std::unordered_map<std::string, double>& variables) {
    auto clean_expression = cleanup_expression(input_expression);
    auto root = create_expression_tree(clean_expression);
    
    print_tree_dfs(root, 0);
    
    auto result = evaluate_tree(root, variables);

    return result;
}