"""
Spectra Python Interop Bridge

Provides Python ctypes bindings to Spectra native acceleration kernel,
data serialization protocols, and reference neural operator verification utilities.
"""

import ctypes
import os
import sys
import math
import struct
from typing import List, Tuple, Optional

# Locate compiled native library
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_NAME = "libspectra_kernel.dylib" if sys.platform == "darwin" else ("spectra_kernel.dll" if sys.platform == "win32" else "libspectra_kernel.so")
_LIB_PATH = os.path.join(_BASE_DIR, "build", _LIB_NAME)

_lib = None
if os.path.exists(_LIB_PATH):
    try:
        _lib = ctypes.CDLL(_LIB_PATH)
        # Function signatures
        _lib.spectra_simd_gemm.argtypes = [
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int64
        ]
        _lib.spectra_simd_gemm.restype = None

        _lib.spectra_chebyshev_eval.argtypes = [
            ctypes.c_int64, ctypes.c_int64,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)
        ]
        _lib.spectra_chebyshev_eval.restype = None
    except Exception as e:
        _lib = None


def is_c_kernel_available() -> bool:
    """Returns True if Spectra native C acceleration shared library is loaded."""
    return _lib is not None


def c_gemm(A: List[List[float]], B: List[List[float]], alpha: float = 1.0, beta: float = 0.0) -> List[List[float]]:
    """
    Performs matrix multiplication C = alpha * A * B + beta * C.
    Falls back to pure Python if native library is not available.
    """
    M = len(A)
    K = len(A[0])
    K2 = len(B)
    N = len(B[0])
    assert K == K2, f"Dimension mismatch: A is {M}x{K}, B is {K2}x{N}"

    if _lib is not None:
        # Convert to column-major 1D flat buffers
        A_flat = [A[i][k] for k in range(K) for i in range(M)]
        B_flat = [B[k][j] for j in range(N) for k in range(K)]
        C_flat = [0.0] * (M * N)

        c_A = (ctypes.c_double * len(A_flat))(*A_flat)
        c_B = (ctypes.c_double * len(B_flat))(*B_flat)
        c_C = (ctypes.c_double * len(C_flat))(*C_flat)

        _lib.spectra_simd_gemm(
            M, N, K, alpha,
            c_A, M,
            c_B, K,
            beta,
            c_C, M
        )

        # Unpack column-major buffer
        C = [[c_C[i + j * M] for j in range(N)] for i in range(M)]
        return C
    else:
        # Pure Python fallback
        C = [[0.0 for _ in range(N)] for _ in range(M)]
        for i in range(M):
            for j in range(N):
                s = sum(A[i][k] * B[k][j] for k in range(K))
                C[i][j] = alpha * s
        return C


def chebyshev_clenshaw(coeffs: List[float], x_points: List[float]) -> List[float]:
    """
    Evaluates sum_{k=0}^{N-1} c_k * T_k(x) across evaluation points.
    """
    n_points = len(x_points)
    deg = len(coeffs)

    if _lib is not None:
        c_x = (ctypes.c_double * n_points)(*x_points)
        c_coeffs = (ctypes.c_double * deg)(*coeffs)
        c_out = (ctypes.c_double * n_points)()

        _lib.spectra_chebyshev_eval(n_points, deg, c_x, c_coeffs, c_out)
        return list(c_out)
    else:
        results = []
        for x in x_points:
            two_x = 2.0 * x
            b_k1 = 0.0
            b_k2 = 0.0
            for k in range(deg - 1, 0, -1):
                b_k = coeffs[k] + two_x * b_k1 - b_k2
                b_k2 = b_k1
                b_k1 = b_k
            res = coeffs[0] + x * b_k1 - b_k2
            results.append(res)
        return results


def dft_1d(signal: List[complex]) -> List[complex]:
    """1D Discrete Fourier Transform reference."""
    N = len(signal)
    out = []
    for k in range(N):
        s = sum(signal[n] * math.e ** (-2j * math.pi * k * n / N) for n in range(N))
        out.append(s)
    return out


def idft_1d(spectrum: List[complex]) -> List[complex]:
    """1D Inverse Discrete Fourier Transform reference."""
    N = len(spectrum)
    out = []
    for n in range(N):
        s = sum(spectrum[k] * math.e ** (2j * math.pi * k * n / N) for k in range(N))
        out.append(s / N)
    return out


def reference_fno_spectral_conv1d(
    x_spatial: List[List[float]],
    weights_real: List[List[List[float]]],
    weights_imag: List[List[List[float]]]
) -> List[List[float]]:
    """
    Reference pure Python 1D Fourier spectral convolution layer:
    x_spatial: (N_spatial, in_channels)
    weights: (out_channels, in_channels, modes)
    Returns: (N_spatial, out_channels)
    """
    N_spatial = len(x_spatial)
    in_c = len(x_spatial[0])
    out_c = len(weights_real)
    modes = len(weights_real[0][0])

    # 1. FFT along spatial dimension for each input channel
    X_fft = []
    for ic in range(in_c):
        sig = [complex(x_spatial[n][ic], 0.0) for n in range(N_spatial)]
        X_fft.append(dft_1d(sig))

    # 2. Multiply truncated modes
    Y_fft = [[0j for _ in range(N_spatial)] for _ in range(out_c)]
    for oc in range(out_c):
        for k in range(min(modes, N_spatial // 2)):
            s = 0j
            for ic in range(in_c):
                w = complex(weights_real[oc][ic][k], weights_imag[oc][ic][k])
                s += w * X_fft[ic][k]
            Y_fft[oc][k] = s

    # 3. IFFT for each output channel
    y_out = [[0.0 for _ in range(out_c)] for _ in range(N_spatial)]
    for oc in range(out_c):
        reconstructed = idft_1d(Y_fft[oc])
        for n in range(N_spatial):
            y_out[n][oc] = reconstructed[n].real

    return y_out
