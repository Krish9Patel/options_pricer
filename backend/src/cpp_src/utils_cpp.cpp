#include "utils_cpp.h"
#include "aad_ops.h"
#include <cmath>
#include <random>

std::vector<double> generate_normal_randoms(int size) {
    /*
       This method is better than just rand() b/c Mersenne Twister algorithm has period 2^19937 - 1  
       This means we have effectively infinite for simulation. This algorithm also has equidistribution properties
       It also passes stringent randomness tests like Dieharder or TestU01
       It is very good for this case, creating a Monte Carlo simulation
    */
    std::random_device rd; // Seeded by rd for non-deterministic seeding
    std::mt19937 gen(rd()); // Mersenne Twister engine for psuedo random num generation
    std::normal_distribution<> d(0.0, 1.0);
    std::vector<double> randoms(size);

    for (int i = 0; i < size; i++) {
        randoms[i] = d(gen);
    }
    
    return randoms;
}

template<typename T> 
std::vector<std::vector<T>> generate_gbm_paths (
    const T& s0,
    const T& r,
    const T& sigma,
    double T_maturity,
    int n_steps,
    int n_sims
) {
    double dt = T_maturity / static_cast<double>(n_steps);
    std::vector<double> z_scores = generate_normal_randoms(n_sims * n_steps);
    std::vector <std::vector<T>> paths(n_sims, std::vector<T>(n_steps + 1));

    T drift = (r - 0.5 * sigma * sigma) * dt;
    T diffusion = sigma * std::sqrt(dt);
    /*
       Fact I learned: ++i is faster than i++ for user-defined types or proxy objects
       This is because ++i modifies in place while i++ makes a copy, modifies, then returns the copy
    */ 
    for(int i = 0; i < n_sims; ++i) {
        paths[i][0] = s0; // Set the initial stock price for each path
        for(int j = 1; j <= n_steps; ++j) {
            double z = z_scores[i * n_steps + (j - 1)]; // Get the random number 
            /* 
               Use the Geometric Brownian Motion formula. 
               S_t = S_{t-1} * exp( drift + diffusion * z )
               All operations here (+, *, exp) are overloaded in aad_ops.h for AD_double
            */
            T exponent = drift + (diffusion * z);
            paths[i][j] = paths[i][j - 1] * exp(exponent); 
        }
    }

    return paths;
}  

template std::vector<std::vector<double>> generate_gbm_paths<double>(
    const double& s0, 
    const double& r, 
    const double& sigma,
    double T_maturity, 
    int n_steps, 
    int n_sims
);

template std::vector<std::vector<AD_double>> generate_gbm_paths<AD_double>(
    const AD_double& s0,
    const AD_double& r, 
    const AD_double& sigma,
    double T_maturity, 
    int n_steps, 
    int n_sims
);