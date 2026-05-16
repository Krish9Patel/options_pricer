#ifndef MATRIX_OPS_AAD_H
#define MATRIX_OPS_AAD_H

#include "aad_types.h"
#include <vector>
#include <numeric> // For std::iota (future matrix creation)
#include <stdexcept> // For exceptions

// Simple matrix class template to work with AD_double 
// For production add dedicated lin alg library like Eigen

template <typename T>
class Matrix {
public:
    int rows;
    int cols;
    std::vector<T> data;
    
    Matrix(int r, int c, T initial_val = T(0.0)) : rows(r), cols(c), data(r * c, initial_val) {
        if (r <= 0 || c <= 0) {
            throw std::invalid_argument("Matrix dimensions must be positive.");
        }
    }

    // Access element -> row-major order
    T& operator()(int r, int c) {
        if (r >= rows || c >= cols || r < 0 || c < 0) {
            throw std::out_of_range("Matrix index out of bounds.");
        }
        return data[r * cols + c];
    }

    const T& operator()(int r, int c) const {
        if (r >= rows || c >= cols || r < 0 || c < 0) {
            throw std::out_of_range("Matrix index out of bounds.");
        }
        return data[r * cols + c];
    }

    // Matrix multiplication (AB)
    template <typename U>
    Matrix<decltype(T() * U())> operator*(const Matrix<U>& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions must match for multiplication.");
        }

        Matrix<decltype(T() * U())> result(rows, other.cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                for (int k = 0; k < cols; ++k) {
                    // result(i, j) += (*this)(i, k) * other(k, j);
                    result(i, j) = result(i, j) + (*this)(i, k) * other(k, j);
                }
            }
        }

        return result;
    }

    // Transpose matrix
    Matrix<T> transpose() const {
        Matrix<T> result(cols, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    // Matrix Inverse 
    Matrix<T> inverse() const {
        if (rows != cols) {
            throw std::invalid_argument("Only square matrices can be inverted.");
        }

        int n = rows;
        Matrix<T> augmented(n, 2 * n);

        // Create augmented matrix [A | I]
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                augmented(i, j) = (*this)(i, j);
            }
            augmented(i, n + i) = T(1.0); // Identity matrix
        }

        // Perform Gaussian elimination
        for (int i = 0; i < n; ++i) {
            // Pivoting
            T pivot = augmented(i, i);
            // Check for zero pivot. For AD_double, we need to check the value.
            // Since we can't easily check type, we rely on operator== being available.
            // If T is AD_double, we need operator==(AD_double, AD_double).
            if (pivot == T(0.0)) {
                throw std::runtime_error("Matrix is singular and cannot be inverted.");
            }
            for (int j = 0; j < 2 * n; ++j) {
                // augmented(i, j) /= pivot;
                augmented(i, j) = augmented(i, j) / pivot;
            }

            // Eliminate other rows
            for (int k = 0; k < n; ++k) {
                if (k != i) {
                    T factor = augmented(k, i);
                    for (int j = 0; j < 2 * n; ++j) {
                        // augmented(k, j) -= factor * augmented(i, j);
                        augmented(k, j) = augmented(k, j) - (factor * augmented(i, j));
                    }
                }
            }
        }

        // Extract the inverse matrix from the augmented matrix
        Matrix<T> inverse(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                inverse(i, j) = augmented(i, n + j);
            }
        }

        return inverse;
    }

}; 

#endif 
