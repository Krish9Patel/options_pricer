#include "aad_smooth.h"
#include <cmath>

// Numerically stable sigmoid for AD_double
// Records on tape: d(sigmoid)/dx = sigmoid * (1 - sigmoid)
AD_double sigmoid(const AD_double& x) {
    double val = x.val();
    double sig_val;
    if (val >= 0.0) {
        double exp_neg = std::exp(-val);
        sig_val = 1.0 / (1.0 + exp_neg);
    } else {
        double exp_pos = std::exp(val);
        sig_val = exp_pos / (1.0 + exp_pos);
    }

    double sig_deriv = sig_val * (1.0 - sig_val);

    AD_double result;
    result.set_val(sig_val);
    // Follow the declared field order: value, adjoint, weight_1, weight_2, parent_1, parent_2
    int idx = static_cast<int>(Tape::instance().push_back(Node{
        sig_val, // Value
        0.0, // Adjoint
        sig_deriv, // Weight 1
        0.0, // Weight 2
        x.tape_index(), // Parent 1
        -1 // Parent 2
    }));
    result.set_tape_index(idx);
    return result;
}

// Smooth max(a, b) using a sigmoid-weighted blend.
// As ε → 0, this approaches the true max.
AD_double smooth_max(const AD_double& a, const AD_double& b, double epsilon) {
    AD_double diff = a - b;
    AD_double p = sigmoid(diff / AD_double(epsilon));
    return p * a + (AD_double(1.0) - p) * b;
}

// Smooth put payoff helper: max(K - S, 0), smoothed.
AD_double smooth_put_payoff(const AD_double& S, double K, double epsilon) {
    return smooth_max(AD_double(K) - S, AD_double(0.0), epsilon);
}
