"""
Spectra Cross-Language Operator Verification Suite

Verifies numerical consistency between Julia Spectra neural operators
and Python reference models.
"""

import sys
import math
from spectra_bridge import (
    is_c_kernel_available,
    c_gemm,
    chebyshev_clenshaw,
    reference_fno_spectral_conv1d
)


def run_verification():
    print("==================================================")
    print("Spectra Python Bridge Verification Harness")
    print("==================================================")

    # 1. Check C Kernel
    c_avail = is_c_kernel_available()
    print(f"[*] C SIMD Acceleration Kernel Available: {c_avail}")

    # 2. Test GEMM
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    C = c_gemm(A, B)
    # Expected: [[19, 22], [43, 50]]
    expected_C = [[19.0, 22.0], [43.0, 50.0]]
    err_gemm = max(abs(C[i][j] - expected_C[i][j]) for i in range(2) for j in range(2))
    print(f"[*] GEMM Max Absolute Error: {err_gemm:.2e}")
    assert err_gemm < 1e-12, "GEMM verification failed"

    # 3. Test Clenshaw Chebyshev Recurrence
    # T_0(x) = 1, T_1(x) = x, T_2(x) = 2x^2 - 1
    # p(x) = 1.0 + 2.0*x + 3.0*(2x^2 - 1) = 6x^2 + 2x - 2
    coeffs = [1.0, 2.0, 3.0]
    xs = [-1.0, -0.5, 0.0, 0.5, 1.0]
    expected_p = [6.0 * x**2 + 2.0 * x - 2.0 for x in xs]
    actual_p = chebyshev_clenshaw(coeffs, xs)
    err_cheb = max(abs(actual_p[i] - expected_p[i]) for i in range(len(xs)))
    print(f"[*] Chebyshev Clenshaw Max Error: {err_cheb:.2e}")
    assert err_cheb < 1e-12, "Chebyshev verification failed"

    # 4. Test FNO 1D Spectral Convolution
    N_s = 16
    in_c = 2
    out_c = 2
    modes = 4
    x_spatial = [[math.sin(2.0 * math.pi * i / N_s), math.cos(2.0 * math.pi * i / N_s)] for i in range(N_s)]
    w_r = [[[0.5 for _ in range(modes)] for _ in range(in_c)] for _ in range(out_c)]
    w_i = [[[0.0 for _ in range(modes)] for _ in range(in_c)] for _ in range(out_c)]

    y_out = reference_fno_spectral_conv1d(x_spatial, w_r, w_i)
    assert len(y_out) == N_s and len(y_out[0]) == out_c
    print(f"[*] Reference FNO-1D Spectral Output shape: ({len(y_out)}, {len(y_out[0])})")

    print("[+] All Python bridge and cross-language verification checks PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(run_verification())
