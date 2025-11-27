#ifndef LSMC_AAD_PRICER_H
#define LSMC_AAD_PRICER_H

#include "aad_types.h"
#include "matrix_ops_aad.h"
#include "utils_cpp.h"
#include <vector>

class LSMCAADPricer {
public:
    // Constructor
    LSMCAADPricer(double s0, double K, double r, double sigma, double T, int n_steps, int n_sims);

    // Main pricing function
    AD_double price();

    // Calculate Greeks (Delta, Vega, Rho)
    // Returns a vector: [Delta, Vega, Rho]
    std::vector<double> calculate_greeks();

private:
    // Raw inputs
    double raw_s0;
    double raw_K;
    double raw_r;
    double raw_sigma;

    // Inputs as AD_double to be recorded on tape
    AD_double s0_ad;
    AD_double K_ad;
    AD_double r_ad;
    AD_double sigma_ad;
    double T_maturity; // Time is usually constant
    int n_steps;
    int n_sims;

    // Helper to compute payoff
    AD_double payoff(const AD_double& S) const;
};

#endif // LSMC_AAD_PRICER_H
