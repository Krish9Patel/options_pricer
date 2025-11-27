#ifndef UTILS_CPP_H
#define UTILS_CPP_H

#include <vector> 
#include <random>
#include "aad_types.h"

std::vector<double> generate_normal_randoms(int size);

template<typename T> 
std::vector<std::vector<T>> generate_gbm_paths(
    const T& s0, // Initial price
    const T& r, // Risk free interest rate 
    const T& sigma, // Volaatility 
    double T_maturity, // Time to maturity
    int n_steps, // Num time steps in the simulation
    int n_sims // Num simulation paths to generate
);

#endif