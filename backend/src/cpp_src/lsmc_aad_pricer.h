#ifndef LSMC_AAD_PRICER_H
#define LSMC_AAD_PRICER_H

#include "aad_types.h"
#include <vector>
#include <random>
#include <cstdint>

class LSMCAADPricer {
public:
    // Configuration
    struct Config {
        int      num_paths;
        int      num_steps;
        double   S0;
        double   K;
        double   r;
        double   sigma;
        double   T;
        int      num_basis;       // number of polynomial basis functions
        double   smooth_epsilon;  // sigmoid smoothing parameter (e.g. 0.01 * K)
        uint64_t rng_seed;      // fixed seed for reproducibility
    };

    LSMCAADPricer(const Config& config);

    // PASS 1: Calibration (pure double, no tape)
    // Populates betas_; must be called before price_and_greeks()
    void calibrate();

    // PASS 2: Valuation with AAD (AD_double, records to tape)
    // Returns the option price; Greeks are written to the out-parameters.
    // Note: theta returned is ∂V/∂T. Conventional Theta is -∂V/∂T.
    double price_and_greeks(
        double& delta,    // ∂V/∂S₀
        double& vega,     // ∂V/∂σ
        double& rho,      // ∂V/∂r
        double& theta     // ∂V/∂T  (negate for conventional Theta)
    );

private:
    Config config_;

    // Stored from Pass 1 — one vector of coefficients per timestep
    // betas_[t] = {β₀, β₁, β₂, ...} for timestep t
    std::vector<std::vector<double>> betas_;

    // RNG state management
    std::mt19937_64 rng_;
    void reset_rng() { rng_.seed(config_.rng_seed); }

    // Helper: evaluate polynomial basis at S given coefficients
    static double evaluate_basis(const std::vector<double>& beta, double S) {
        double result = 0.0;
        double basis  = 1.0;
        for (size_t k = 0; k < beta.size(); ++k) {
            result += beta[k] * basis;
            basis  *= S;
        }
        return result;
    }
};

#endif // LSMC_AAD_PRICER_H
