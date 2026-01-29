/*
 * Spectra High-Performance C Native Acceleration Kernel
 *
 * SIMD-vectorized, cache-blocked numerical primitives for spectral operators,
 * sheaf Laplacian diffusion, and symplectic dynamics.
 */

#ifndef SPECTRA_KERNEL_H
#define SPECTRA_KERNEL_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Cache-blocked matrix multiplication: C = alpha * A * B + beta * C
 * A: (M x K), B: (K x N), C: (M x N) in column-major order.
 */
void spectra_simd_gemm(
    int64_t M, int64_t N, int64_t K,
    double alpha,
    const double *A, int64_t lda,
    const double *B, int64_t ldb,
    double beta,
    double *C, int64_t ldc
);

/*
 * Fused spectral tensor contraction for neural operators:
 * Y(m, oc, ic) = sum_k W(oc, ic, k) * X(m, ic, k)
 */
void spectra_simd_tensor_contract(
    int64_t n_modes,
    int64_t in_channels,
    int64_t out_channels,
    const double *weights, // (out_channels x in_channels x n_modes)
    const double *X_real,  // (n_modes x in_channels)
    const double *X_imag,  // (n_modes x in_channels)
    double *Y_real,        // (n_modes x out_channels)
    double *Y_imag         // (n_modes x out_channels)
);

/*
 * Vectorized Clenshaw-Smith recurrence for Chebyshev polynomial sum:
 * p(x) = sum_{k=0}^{N-1} c_k * T_k(x)
 */
void spectra_chebyshev_eval(
    int64_t n_points,
    int64_t degree,
    const double *x_points,
    const double *coeffs,
    double *out_results
);

/*
 * Fused symplectic leapfrog momentum-coordinate update step:
 * p_{n+1/2} = p_n - 0.5 * dt * grad_V(q_n)
 * q_{n+1}   = q_n + dt * M^{-1} * p_{n+1/2}
 * p_{n+1}   = p_{n+1/2} - 0.5 * dt * grad_V(q_{n+1})
 */
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
);

/*
 * Sparse sheaf diffusion step:
 * x_{t+1} = x_t - tau * L * x_t
 * L is given in Compressed Sparse Column (CSC) format.
 */
void spectra_sheaf_diffuse(
    int64_t n,
    double tau,
    const int64_t *colptr, // 1-based or 0-based handled internally
    const int64_t *rowval,
    const double *nzval,
    const double *x_in,
    double *x_out
);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRA_KERNEL_H */
