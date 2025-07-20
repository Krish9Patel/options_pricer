#ifndef AAD_OPS_H
#define AAD_OPS_H

#include "aad_types.h"
#include <cmath>

enum Optype {
    OP_NONE = 0,
    OP_ADD,
    OP_SUB, 
    OP_MUL,
    OP_DIV,
    OP_NEG, // unary minus
    OP_EXP,
    OP_LOG,
    OP_SQRT,
    OP_SIN,
    OP_COS,
    OP_TAN,
    OP_COS,
    OP_TAN,
    // I think thats good enough for now, can add more later
};


// Overloaded operators for AD_double
AD_double operator+(const AD_double& lhs, const AD_double& rhs);
AD_double operator-(const AD_double& lhs, const AD_double& rhs);
AD_double operator*(const AD_double& lhs, const AD_double& rhs);
AD_double operator/(const AD_double& lhs, const AD_double& rhs);
AD_double operator-(const AD_double& val); // Unary minus

// Scalar 
AD_double operator+(const AD_double& lhs, double rhs);
AD_double operator+(double lhs, const AD_double& rhs);
AD_double operator-(const AD_double& lhs, double rhs);
AD_double operator-(double lhs, const AD_double& rhs);
AD_double operator*(const AD_double& lhs, double rhs);
AD_double operator*(double lhs, const AD_double& rhs);
AD_double operator/(const AD_double& lhs, double rhs);
AD_double operator/(double lhs, const AD_double& rhs);

// Math functions
AD_double exp(const AD_double& val);
AD_double log(const AD_double& val);
AD_double sqrt(const AD_double& val);
AD_double sin(const AD_double& val);
AD_double cos(const AD_double& val);
AD_double tan(const AD_double& val);

#endif