#include "aad_ops.h"

std::shared_ptr<Tape> curr_tape = nullptr;

void initialize_aad_tape() {
    if (!curr_tape) {
        curr_tape = std::make_shared<Tape>();
    } else {
        curr_tape -> clear(); // Clear existing tape if initialized
    }
}

// Perform reverse pass to compute adjoints
void reverse_pass(int start_idx) {
    if (!curr_tape || curr_tape -> nodes.empty()) {
        return;
    }

    if (start_idx != -1 && start_idx < curr_tape -> nodes.size()) {
        curr_tape -> nodes[start_idx].adjoint = 1.0; // Initialize adjoint of the output node
    } else {
        curr_tape -> nodes.back().adjoint = 1.0; // Default to last node if no start index provided
    }

    for (int i = curr_tape -> nodes.size() - 1; i >= 0; --i) {
        Node& node = curr_tape -> nodes[i];

        if (node.adjoint == 0.0 && i != start_idx) {
            continue; // Skip if adjoint = 0 
        }

        for (size_t p_idx = 0; p_idx < node.parents.size(); ++p_idx) {
            int parent_tape_idx = node.parents[p_idx];
            double weight = node.weights[p_idx];
            curr_tape -> nodes[parent_tape_idx].adjoint += node.adjoint * weight;
        }
    }
}

// Operator overloads 
AD_double operator+(const AD_double& lhs, const AD_double& rhs) {
    double result = lhs.val() + rhs.val();
    int new_idx = curr_tape -> record(result, OP_ADD, {lhs.tape_idx, rhs.tape_idx}, {1.0, 1.0}); // d(a+b)/da = 1, d(a+b)/db = 1
    return AD_double(new_idx);
}

AD_double operator-(const AD_double& lhs, const AD_double& rhs) {
    double result = lhs.val() - rhs.val();
    int new_idx = curr_tape -> record(result, OP_SUB, {lhs.tape_idx, rhs.tape_idx}, {1.0, -1.0}); // d(a-b)/da = 1, d(a-b)/db = -1
    return AD_double(new_idx);
}

AD_double operator*(const AD_double& lhs, const AD_double& rhs) {
    double result = lhs.val() * rhs.val();
    int new_idx = curr_tape -> record(result, OP_MUL, {lhs.tape_idx, rhs.tape_idx}, {rhs.val(), lhs.val()}); // d(ab)/da = b, d(ab)/db = a
    return AD_double(new_idx);
}

AD_double operator/(const AD_double& lhs, const AD_double& rhs) { 
    double result = lhs.val() / rhs.val();
    int new_idx = curr_tape -> record(result, OP_DIV, {lhs.tape_idx, rhs.tape_idx}, {1.0 / rhs.val(), -lhs.val() / (rhs.val() * rhs.val())}); // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    return AD_double(new_idx);
}

// Unary minus
AD_double operator-(const AD_double& val) {
    double result = -val.val();
    int new_idx = curr_tape -> record(result, OP_NEG, {val.tape_idx}, {-1.0}); // d(-a)/da = -1
    return AD_double(new_idx);
}

// Scalar operations
AD_double operator+(const AD_double& lhs, double rhs) {
    return lhs + AD_double(rhs); // Convert to AD_double 
}

AD_double operator+(double lhs, const AD_double& rhs) {
    return AD_double(lhs) + rhs; 
}

AD_double operator-(const AD_double& lhs, double rhs) {
    return lhs - AD_double(rhs); // Convert to AD_double 
}

AD_double operator-(double lhs, const AD_double& rhs) {
    return AD_double(lhs) - rhs; 
}

AD_double operator*(const AD_double& lhs, double rhs) {
    return lhs * AD_double(rhs); // Convert to AD_double 
}

AD_double operator*(double lhs, const AD_double& rhs) {
    return AD_double(lhs) * rhs; 
}

AD_double operator/(const AD_double& lhs, double rhs) {
    return lhs / AD_double(rhs); // Convert to AD_double 
}

AD_double operator/(double lhs, const AD_double& rhs) {
    return AD_double(lhs) / rhs; 
}

// Comparison
bool operator==(const AD_double& lhs, const AD_double& rhs) {
    return lhs.val() == rhs.val();
}
bool operator==(const AD_double& lhs, double rhs) {
    return lhs.val() == rhs;
}
bool operator==(double lhs, const AD_double& rhs) {
    return lhs == rhs.val();
}
bool operator!=(const AD_double& lhs, const AD_double& rhs) {
    return !(lhs == rhs);
}
bool operator!=(const AD_double& lhs, double rhs) {
    return !(lhs == rhs);
}
bool operator!=(double lhs, const AD_double& rhs) {
    return !(lhs == rhs);
}

// Math functions 
AD_double exp(const AD_double& val) {
    double result = std::exp(val.val());
    int new_idx = curr_tape -> record(result, OP_EXP, {val.tape_idx}, {result}); // d(exp(a))/da = exp(a)
    return AD_double(new_idx);
}

AD_double log(const AD_double& val) {
    double result = std::log(val.val());
    int new_idx = curr_tape -> record(result, OP_LOG, {val.tape_idx}, {1.0 / val.val()}); // d(log(a))/da = 1/a
    return AD_double(new_idx);
}

AD_double sqrt(const AD_double& val) {
    double result = std::sqrt(val.val());
    int new_idx = curr_tape -> record(result, OP_SQRT, {val.tape_idx}, {(0.5 / result)}); // d(sqrt(a))/da = 0.5 / sqrt(a))
    return AD_double(new_idx);
}

AD_double sin(const AD_double& val) {
    double result = std::sin(val.val());
    int new_idx = curr_tape -> record(result, OP_SIN, {val.tape_idx}, {std::cos(val.val())}); // d(sin(a))/da = cos(a)
    return AD_double(new_idx);
}

AD_double cos(const AD_double& val) {
    double result = std::cos(val.val());
    int new_idx = curr_tape -> record(result, OP_COS, {val.tape_idx}, {-std::sin(val.val())}); // d(cos(a))/da = -sin(a)
    return AD_double(new_idx);
}

AD_double tan(const AD_double& val) {
    double result = std::tan(val.val());
    int new_idx = curr_tape -> record(result, OP_TAN, {val.tape_idx}, {1.0 + result * result}); // d(tan(a))/da = sec^2(a)
    return AD_double(new_idx);
}
