#include "aad_ops.h"

// Helper: push a binary node and wrap as AD_double
static AD_double make_binary_node(double value, int p1, double w1, int p2, double w2) {
    AD_double result;
    result.set_val(value);
    int idx = static_cast<int>(Tape::instance().push_back(Node{value, 0.0, w1, w2, p1, p2}));
    result.set_tape_index(idx);
    return result;
}

// Unary helper
static AD_double make_unary_node(double value, int p, double w) {
    AD_double result;
    result.set_val(value);
    int idx = static_cast<int>(Tape::instance().push_back(Node{value, 0.0, w, 0.0, p, -1}));
    result.set_tape_index(idx);
    return result;
}


// Addition with constant fast paths
AD_double operator+(const AD_double& lhs, const AD_double& rhs) {
    int p1 = lhs.tape_index();
    int p2 = rhs.tape_index();
    if (p1 < 0 && p2 < 0) return AD_double(lhs.val() + rhs.val());  // const + const
    return make_binary_node(lhs.val() + rhs.val(), p1, 1.0, p2, 1.0);
}

// Subtraction
AD_double operator-(const AD_double& lhs, const AD_double& rhs) {
    int p1 = lhs.tape_index();
    int p2 = rhs.tape_index();
    if (p1 < 0 && p2 < 0) return AD_double(lhs.val() - rhs.val());
    return make_binary_node(lhs.val() - rhs.val(), p1, 1.0, p2, -1.0);
}

// Multiplication: ∂(a·b)/∂a = b, ∂(a·b)/∂b = a
AD_double operator*(const AD_double& lhs, const AD_double& rhs) {
    int p1 = lhs.tape_index();
    int p2 = rhs.tape_index();
    if (p1 < 0 && p2 < 0) return AD_double(lhs.val() * rhs.val());
    return make_binary_node(lhs.val() * rhs.val(), p1, rhs.val(), p2, lhs.val());
}

// Division: ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b²
AD_double operator/(const AD_double& lhs, const AD_double& rhs) {
    int p1 = lhs.tape_index();
    int p2 = rhs.tape_index();
    double inv_b = 1.0 / rhs.val();
    if (p1 < 0 && p2 < 0) return AD_double(lhs.val() * inv_b);
    return make_binary_node(lhs.val() * inv_b, p1, inv_b, p2, -lhs.val() * inv_b * inv_b);
}

// Unary minus
AD_double operator-(const AD_double& val) {
    int p = val.tape_index();
    if (p < 0) return AD_double(-val.val());
    return make_unary_node(-val.val(), p, -1.0);
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
bool operator<(const AD_double& lhs, const AD_double& rhs) {
    return lhs.val() < rhs.val();
}
bool operator<(const AD_double& lhs, double rhs) {
    return lhs.val() < rhs;
}
bool operator<(double lhs, const AD_double& rhs) {
    return lhs < rhs.val();
}
bool operator>(const AD_double& lhs, const AD_double& rhs) {
    return lhs.val() > rhs.val();
}
bool operator>(const AD_double& lhs, double rhs) {
    return lhs.val() > rhs;
}
bool operator>(double lhs, const AD_double& rhs) {
    return lhs > rhs.val();
}

// Math functions 
AD_double exp(const AD_double& x) {
    double v = std::exp(x.val());
    if (x.tape_index() < 0) return AD_double(v);
    return make_unary_node(v, x.tape_index(), v);  // d(exp)/dx = exp
}

AD_double log(const AD_double& x) {
    double v = std::log(x.val());
    if (x.tape_index() < 0) return AD_double(v);
    return make_unary_node(v, x.tape_index(), 1.0 / x.val());
}

AD_double sqrt(const AD_double& x) {
    double v = std::sqrt(x.val());
    if (x.tape_index() < 0) return AD_double(v);
    return make_unary_node(v, x.tape_index(), 0.5 / v);
}

AD_double sin(const AD_double& x) {
    double v = std::sin(x.val());
    if (x.tape_index() < 0) return AD_double(v);
    return make_unary_node(v, x.tape_index(), std::cos(x.val()));
}

AD_double cos(const AD_double& x) {
    double v = std::cos(x.val());
    if (x.tape_index() < 0) return AD_double(v);
    return make_unary_node(v, x.tape_index(), -std::sin(x.val()));
}

AD_double tan(const AD_double& x) {
    double v = std::tan(x.val());
    if (x.tape_index() < 0) return AD_double(v);
    return make_unary_node(v, x.tape_index(), 1.0 + v * v);
}
