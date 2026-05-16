#ifndef AAD_SMOOTH_H
#define AAD_SMOOTH_H

#include "aad_types.h"
#include "aad_ops.h"

// Numerically stable sigmoid for AD_double
AD_double sigmoid(const AD_double& x);

// Smooth max(a, b) using a sigmoid-weighted blend.
AD_double smooth_max(const AD_double& a, const AD_double& b, double epsilon);

// Smooth put payoff helper: max(K - S, 0), smoothed.
AD_double smooth_put_payoff(const AD_double& S, double K, double epsilon);

#endif 
