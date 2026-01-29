/*
 * Spectra High-Performance C Native Acceleration Kernel
 *
 * Implements cache-blocked GEMM, tensor contraction, Clenshaw Chebyshev recurrence,
 * symplectic momentum integration, and sparse sheaf diffusion.
 */

#include "spectra_kernel.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define BLOCK_SIZE 64

void spectra_simd_gemm(
    int64_t M, int64_t N, int64_t K,
    double alpha,
    const double *A, int64_t lda,
    const double *B, int64_t ldb,
    double beta,
    double *C, int64_t ldc
) {
    /* Scale C by beta */
    if (beta == 0.0) {
        for (int64_t j = 0; j < N; ++j) {
            for (int64_t i = 0; i < M; ++i) {
                C[i + j * ldc] = 0.0;
            }
        }
    } else if (beta != 1.0) {
        for (int64_t j = 0; j < N; ++j) {
            for (int64_t i = 0; i < M; ++i) {
                C[i + j * ldc] *= beta;
            }
        }
    }

    /* Cache-blocked loop hierarchy */
    for (int64_t jj = 0; jj < N; jj += BLOCK_SIZE) {
        int64_t j_end = (jj + BLOCK_SIZE < N) ? (jj + BLOCK_SIZE) : N;
        for (int64_t kk = 0; kk < K; kk += BLOCK_SIZE) {
            int64_t k_end = (kk + BLOCK_SIZE < K) ? (kk + BLOCK_SIZE) : K;
            for (int64_t ii = 0; ii < M; ii += BLOCK_SIZE) {
                int64_t i_end = (ii + BLOCK_SIZE < M) ? (ii + BLOCK_SIZE) : M;

                for (int64_t j = jj; j < j_end; ++j) {
                    for (int64_t k = kk; k < k_end; ++k) {
                        double b_val = alpha * B[k + j * ldb];
                        #pragma clang loop vectorize(enable)
                        for (int64_t i = ii; i < i_end; ++i) {
                            C[i + j * ldc] += A[i + k * lda] * b_val;
                        }
                    }
                }
            }
        }
    }
}

void spectra_simd_tensor_contract(
    int64_t n_modes,
    int64_t in_channels,
    int64_t out_channels,
    const double *weights, // (out_channels x in_channels x n_modes)
    const double *X_real,  // (n_modes x in_channels)
    const double *X_imag,  // (n_modes x in_channels)
    double *Y_real,        // (n_modes x out_channels)
    double *Y_imag         // (n_modes x out_channels)
) {
    /* Initialize Y to zero */
    memset(Y_real, 0, sizeof(double) * n_modes * out_channels);
    if (X_imag != NULL && Y_imag != NULL) {
        memset(Y_imag, 0, sizeof(double) * n_modes * out_channels);
    }

    for (int64_t m = 0; m < n_modes; ++m) {
        for (int64_t oc = 0; oc < out_channels; ++oc) {
            double sum_real = 0.0;
            double sum_imag = 0.0;

            for (int64_t ic = 0; ic < in_channels; ++ic) {
                /* weight index: oc + ic * out_channels + m * (out_channels * in_channels) */
                int64_t w_idx = oc + ic * out_channels + m * (out_channels * in_channels);
                double w = weights[w_idx];

                /* X index: m + ic * n_modes */
                int64_t x_idx = m + ic * n_modes;
                sum_real += w * X_real[x_idx];
                if (X_imag != NULL) {
                    sum_imag += w * X_imag[x_idx];
                }
            }

            int64_t y_idx = m + oc * n_modes;
            Y_real[y_idx] = sum_real;
            if (Y_imag != NULL) {
                Y_imag[y_idx] = sum_imag;
            }
        }
    }
}

void spectra_chebyshev_eval(
    int64_t n_points,
    int64_t degree,
    const double *x_points,
    const double *coeffs,
    double *out_results
) {
    if (degree <= 0) {
        memset(out_results, 0, sizeof(double) * n_points);
        return;
    }

    for (int64_t p = 0; p < n_points; ++p) {
        double x = x_points[p];
        double two_x = 2.0 * x;

        /* Clenshaw recurrence backwards from degree-1 */
        double b_k1 = 0.0;
        double b_k2 = 0.0;

        for (int64_t k = degree - 1; k >= 1; --k) {
            double b_k = coeffs[k] + two_x * b_k1 - b_k2;
            b_k2 = b_k1;
            b_k1 = b_k;
        }

        /* Final step for T_0(x) = 1 */
        out_results[p] = coeffs[0] + x * b_k1 - b_k2;
    }
}

void spectra_symplectic_step(
    int64_t dim,
    double dt,
    const double *q_in,
    const double *p_in,
    const double *grad_V_n,
    const double *grad_V_next,
    const double *inv_mass,
    double *q_out,
    double *p_out
) {
    double half_dt = 0.5 * dt;

    for (int64_t i = 0; i < dim; ++i) {
        double inv_m = (inv_mass != NULL) ? inv_mass[i] : 1.0;

        /* Half-step momentum update */
        double p_half = p_in[i] - half_dt * grad_V_n[i];

        /* Full-step coordinate update */
        q_out[i] = q_in[i] + dt * inv_m * p_half;

        /* Final half-step momentum update */
        p_out[i] = p_half - half_dt * grad_V_next[i];
    }
}

void spectra_sheaf_diffuse(
    int64_t n,
    double tau,
    const int64_t *colptr, // 1-based indices from Julia CSC
    const int64_t *rowval, // 1-based indices from Julia CSC
    const double *nzval,
    const double *x_in,
    double *x_out
) {
    /* First copy x_in to x_out */
    memcpy(x_out, x_in, sizeof(double) * n);

    /* Compute y = L * x_in and subtract tau * y */
    for (int64_t col = 0; col < n; ++col) {
        int64_t start = colptr[col] - 1;     // convert 1-based to 0-based
        int64_t end   = colptr[col + 1] - 1; // convert 1-based to 0-based
        double x_val = x_in[col];

        for (int64_t idx = start; idx < end; ++idx) {
            int64_t row = rowval[idx] - 1;   // convert 1-based to 0-based
            double val = nzval[idx];
            x_out[row] -= tau * val * x_val;
        }
    }
}
