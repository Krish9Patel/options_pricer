#include "lsmc_aad_pricer.h"
#include "aad_ops.h"
#include "aad_smooth.h"
#include "matrix_ops_aad.h"
#include <cmath>
#include <algorithm>
#include <iostream>

LSMCAADPricer::LSMCAADPricer(const Config& config)
    : config_(config) {}

void LSMCAADPricer::calibrate() {
    reset_rng();
    const int N = config_.num_paths;
    const int M = config_.num_steps;
    const double dt = config_.T / M;
    const double df = std::exp(-config_.r * dt);

    // 1. Simulate all paths (standard double)
    std::vector<std::vector<double>> paths(N, std::vector<double>(M + 1));
    std::normal_distribution<double> normal(0.0, 1.0);

    const double drift_term  = (config_.r - 0.5 * config_.sigma * config_.sigma) * dt;
    const double diffuse_term = config_.sigma * std::sqrt(dt);

    for (int i = 0; i < N; ++i) {
        paths[i][0] = config_.S0;
        for (int t = 1; t <= M; ++t) {
            double z = normal(rng_);
            paths[i][t] = paths[i][t-1] * std::exp(drift_term + diffuse_term * z);
        }
    }

    // 2. Initialize cash flows at maturity
    std::vector<double> cash_flows(N);
    for (int i = 0; i < N; ++i) {
        cash_flows[i] = std::max(config_.K - paths[i][M], 0.0); // Put payoff
    }

    // 3. Backward induction - store betas at each step
    betas_.assign(M + 1, std::vector<double>(config_.num_basis, 0.0));

    for (int t = M - 1; t >= 1; --t) {
        std::vector<int> itm_indices;
        itm_indices.reserve(N);
        for (int i = 0; i < N; ++i) {
            if (config_.K - paths[i][t] > 0.0) {
                itm_indices.push_back(i);
            }
        }

        const int K_basis = config_.num_basis;
        if (static_cast<int>(itm_indices.size()) >= K_basis) {
            const int n_itm = itm_indices.size();

            Matrix<double> X(n_itm, K_basis);
            Matrix<double> Y(n_itm, 1);

            for (int j = 0; j < n_itm; ++j) {
                int idx = itm_indices[j];
                double S = paths[idx][t];
                double basis = 1.0;
                for (int k = 0; k < K_basis; ++k) {
                    X(j, k) = basis;
                    basis *= S;
                }
                Y(j, 0) = cash_flows[idx] * df; 
            }

            try {
                Matrix<double> XT = X.transpose();
                Matrix<double> XTX = XT * X;
                Matrix<double> XTX_inv = XTX.inverse();
                Matrix<double> Beta = (XTX_inv * XT) * Y;

                for (int k = 0; k < K_basis; ++k) {
                    betas_[t][k] = Beta(k, 0);
                }
            } catch (const std::exception& e) {
                // Keep betas as 0.0 on singular matrix
            }
        }

        // MAJOR FIX: discount every path first.
        for (int i = 0; i < N; ++i) {
            cash_flows[i] *= df;
        }

        // Overwrite ITM paths if exercise is optimal
        for (int idx : itm_indices) {
            double S = paths[idx][t];
            double exercise = config_.K - S;
            double continuation = evaluate_basis(betas_[t], S);
            if (exercise > continuation) {
                cash_flows[idx] = exercise;
            }
        }
    }
}

double LSMCAADPricer::price_and_greeks(
    double& delta, double& vega, double& rho, double& theta)
{
    Tape::instance().clear();
    Tape::instance().reserve(
        static_cast<size_t>(config_.num_paths) * config_.num_steps * 15
    );

    reset_rng(); 

    const int N = config_.num_paths;
    const int M = config_.num_steps;

    AD_double S0_val(config_.S0);
    AD_double sigma_val(config_.sigma);
    AD_double r_val(config_.r);
    AD_double T_val(config_.T);
    S0_val.mark_as_input();     
    sigma_val.mark_as_input();  
    r_val.mark_as_input();      
    T_val.mark_as_input();      

    AD_double M_const(static_cast<double>(M));
    AD_double dt_val = T_val / M_const;
    AD_double sqrt_dt = sqrt(dt_val);
    AD_double half(0.5);
    AD_double half_sigma_sq_dt = half * sigma_val * sigma_val * dt_val;
    AD_double drift_term = r_val * dt_val - half_sigma_sq_dt;
    AD_double diffuse_coeff = sigma_val * sqrt_dt;
    AD_double df_val = exp(AD_double(-1.0) * r_val * dt_val);
    AD_double K_val(config_.K);
    AD_double one(1.0);
    AD_double eps_val(config_.smooth_epsilon);
    AD_double inv_N(1.0 / static_cast<double>(N));

    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<AD_double> cash_flows(N);

    for (int i = 0; i < N; ++i) {
        std::vector<AD_double> path_S(M + 1);
        path_S[0] = S0_val;

        for (int t = 1; t <= M; ++t) {
            double z = normal(rng_);
            AD_double z_val(z);      
            path_S[t] = path_S[t-1] * exp(drift_term + diffuse_coeff * z_val);
        }

        cash_flows[i] = smooth_max(K_val - path_S[M], AD_double(0.0), config_.smooth_epsilon);

        for (int t = M - 1; t >= 1; --t) {
            AD_double exercise_val = smooth_max(K_val - path_S[t], AD_double(0.0), config_.smooth_epsilon);

            AD_double continuation(0.0);
            AD_double basis(1.0);
            for (int k = 0; k < config_.num_basis; ++k) {
                continuation = continuation + AD_double(betas_[t][k]) * basis;
                basis = basis * path_S[t];
            }

            AD_double diff = exercise_val - continuation;
            AD_double prob_exercise = sigmoid(diff / eps_val);

            AD_double discounted_future = cash_flows[i] * df_val;

            cash_flows[i] = prob_exercise * exercise_val + (one - prob_exercise) * discounted_future;
        }
    }

    AD_double price(0.0);
    for (int i = 0; i < N; ++i) {
        price = price + cash_flows[i];
    }
    price = price * inv_N * df_val;

    price.propagate_adjoint(1.0);

    delta = S0_val.adjoint();
    vega = sigma_val.adjoint();
    rho = r_val.adjoint();
    theta = T_val.adjoint();

    return price.val();
}
