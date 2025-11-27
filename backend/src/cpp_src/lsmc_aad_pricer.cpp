#include "lsmc_aad_pricer.h"
#include "aad_ops.h"
#include <cmath>
#include <algorithm>
#include <iostream>

LSMCAADPricer::LSMCAADPricer(double s0, double K, double r, double sigma, double T, int n_steps, int n_sims)
    : raw_s0(s0), raw_K(K), raw_r(r), raw_sigma(sigma), T_maturity(T), n_steps(n_steps), n_sims(n_sims) {}

AD_double LSMCAADPricer::payoff(const AD_double& S) const {
    /* American Put Option Payoff: max(K - S, 0)
       We use a conditional on the primal value to determine the operation
       This is valid for AAD as the derivative is defined almost everywhere
    */
    if (S.val() < K_ad.val()) {
        return K_ad - S;
    } else {
        return AD_double(0.0);
    }
}

AD_double LSMCAADPricer::price() {
    // 1. Initialize Tape
    initialize_aad_tape();

    // 2. Register inputs on the tape
    s0_ad = AD_double(raw_s0);
    K_ad = AD_double(raw_K);
    r_ad = AD_double(raw_r);
    sigma_ad = AD_double(raw_sigma);

    // 3. Generate GBM Paths
    // paths[sim][step]
    std::vector<std::vector<AD_double>> paths = generate_gbm_paths<AD_double>(
        s0_ad, r_ad, sigma_ad, T_maturity, n_steps, n_sims
    );

    double dt = T_maturity / static_cast<double>(n_steps);
    AD_double df = exp(-r_ad * dt);

    // 4. Initialize Cash Flows at Maturity (T)
    std::vector<AD_double> cash_flows(n_sims);
    for (int i = 0; i < n_sims; ++i) {
        cash_flows[i] = payoff(paths[i][n_steps]);
    }

    // 5. Backward Induction
    for (int t = n_steps - 1; t > 0; --t) {
        // Discount cash flows from t+1 to t
        for (int i = 0; i < n_sims; ++i) {
            cash_flows[i] = cash_flows[i] * df;
        }

        // Identify In-The-Money (ITM) paths
        std::vector<int> itm_indices;
        for (int i = 0; i < n_sims; ++i) {
            // Check if ITM: Put option -> S < K
            if (paths[i][t].val() < K_ad.val()) {
                itm_indices.push_back(i);
            }
        }

        // If not enough paths are ITM, we can't do regression reliably
        if (itm_indices.size() < 3) {
            continue; // Skip regression, keep discounted continuation values
        }

        // Build Regression Matrices
        // X: Basis functions [1, S, S^2]
        // Y: Discounted continuation values (current cash_flows)
        int n_itm = itm_indices.size();
        Matrix<AD_double> X(n_itm, 3);
        Matrix<AD_double> Y(n_itm, 1);

        for (int k = 0; k < n_itm; ++k) {
            int sim_idx = itm_indices[k];
            AD_double S = paths[sim_idx][t];
            
            X(k, 0) = AD_double(1.0);
            X(k, 1) = S;
            X(k, 2) = S * S;

            Y(k, 0) = cash_flows[sim_idx];
        }

        // Solve for Beta: B = (X'X)^-1 X'Y
        try {
            Matrix<AD_double> XT = X.transpose();
            Matrix<AD_double> XTX = XT * X;
            Matrix<AD_double> XTX_inv = XTX.inverse();
            Matrix<AD_double> Beta = (XTX_inv * XT) * Y;

            // Update Cash Flows
            for (int k = 0; k < n_itm; ++k) {
                int sim_idx = itm_indices[k];
                AD_double S = paths[sim_idx][t];
                
                // Estimated continuation value
                AD_double continuation_val = Beta(0, 0) + Beta(1, 0) * S + Beta(2, 0) * S * S;
                AD_double exercise_val = payoff(S);

                // Decision: Exercise if Exercise Value > Continuation Value
                // Note: Using primal value for decision
                if (exercise_val.val() > continuation_val.val()) {
                    cash_flows[sim_idx] = exercise_val;
                }
            }
        } catch (const std::exception& e) {
            // If matrix inversion fails (singular), skip regression for this step
            // In a robust system, we might use SVD or QR
            // For now, we just proceed with existing cash flows
        }
    }

    // 6. Discount to Time 0
    for (int i = 0; i < n_sims; ++i) {
        cash_flows[i] = cash_flows[i] * df;
    }

    // 7. Average
    AD_double sum(0.0);
    for (const auto& cf : cash_flows) {
        sum = sum + cf;
    }

    return sum / static_cast<double>(n_sims);
}

std::vector<double> LSMCAADPricer::calculate_greeks() {
    AD_double price_ad = price();
    
    // Seed adjoint of the result
    price_ad.set_adjoint(1.0);

    // Run reverse pass
    // We need to know where to start. 
    // Usually reverse_pass() takes the index of the result node.
    reverse_pass(price_ad.tape_idx);

    // Extract adjoints
    double delta = s0_ad.adj();
    double vega = sigma_ad.adj();
    double rho = r_ad.adj();

    return {delta, vega, rho};
}
