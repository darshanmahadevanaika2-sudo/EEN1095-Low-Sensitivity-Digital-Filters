"""
Core filter design module for the multi-precision evaluation framework.
Designs IIR and FIR filters using SciPy and converts coefficients
to target precision levels (float16, float32, float64, mpmath).

"""

import numpy as np
from scipy import signal
import mpmath
import warnings
warnings.filterwarnings('ignore')

# Precision levels
PRECISION_LEVELS = {
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
    'mpmath':  None,          # handled separately via mpmath
}

MPMATH_DPS = 50               # 50 decimal digits of precision


def set_mpmath_precision():
    """Set mpmath to 50 decimal digit precision."""
    mpmath.mp.dps = MPMATH_DPS


# IIR Filter Design
def design_iir_filter(filter_type, order, cutoff=0.3, btype='low',
                       transition_bw=0.02, rp=1.0, rs=40.0):
    """
    Design an IIR filter using SciPy.

    Parameters
    ----------
    filter_type : str
        One of 'butter', 'cheby1', 'cheby2', 'ellip'
    order : int
        Filter order (10, 20, or 30)
    cutoff : float
        Normalised cutoff frequency (0 to 1)
    btype : str
        Filter band type: 'low', 'high', 'band'
    transition_bw : float
        Normalised transition bandwidth (default 0.02)
    rp : float
        Maximum ripple in passband (dB) for cheby1/ellip
    rs : float
        Minimum attenuation in stopband (dB) for cheby2/ellip

    Returns
    -------
    sos : ndarray
        Second-order sections representation (float64)
    b, a : ndarray
        Transfer function coefficients (float64)
    """
    set_mpmath_precision()

    # Adjust cutoff based on transition bandwidth
    wp = cutoff
    ws = cutoff + transition_bw

    if filter_type == 'butter':
        sos = signal.butter(order, wp, btype=btype, output='sos')
        b, a = signal.butter(order, wp, btype=btype, output='ba')
    elif filter_type == 'cheby1':
        sos = signal.cheby1(order, rp, wp, btype=btype, output='sos')
        b, a = signal.cheby1(order, rp, wp, btype=btype, output='ba')
    elif filter_type == 'cheby2':
        sos = signal.cheby2(order, rs, ws, btype=btype, output='sos')
        b, a = signal.cheby2(order, rs, ws, btype=btype, output='ba')
    elif filter_type == 'ellip':
        sos = signal.ellip(order, rp, rs, wp, btype=btype, output='sos')
        b, a = signal.ellip(order, rp, rs, wp, btype=btype, output='ba')
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return sos, b, a


# FIR Filter Design
def design_fir_filter(num_taps, cutoff=0.3, window='kaiser', beta=8.6):
    """
    Design a FIR filter using the Kaiser window method.

    Parameters
    ----------
    num_taps : int
        Number of filter taps (100, 200, or 300)
    cutoff : float
        Normalised cutoff frequency (0 to 1)
    window : str
        Window type (default 'kaiser')
    beta : float
        Kaiser window beta parameter (default 8.6 gives ~60dB stopband attenuation)

    Returns
    -------
    h : ndarray
        FIR filter coefficients (float64)
    """
    if window == 'kaiser':
        h = signal.firwin(num_taps, cutoff, window=('kaiser', beta))
    else:
        h = signal.firwin(num_taps, cutoff, window=window)
    return h


# Precision Conversion
def convert_to_precision(coeffs, precision):
    """
    Convert filter coefficients to the target precision level.

    Parameters
    ----------
    coeffs : ndarray
        Float64 coefficients from SciPy design
    precision : str
        Target precision: 'float16', 'float32', 'float64', 'mpmath'

    Returns
    -------
    converted : ndarray or list of mpmath.mpf
        Coefficients at target precision
    """
    set_mpmath_precision()

    if precision == 'float16':
        return coeffs.astype(np.float16)
    elif precision == 'float32':
        return coeffs.astype(np.float32)
    elif precision == 'float64':
        return coeffs.astype(np.float64)
    elif precision == 'mpmath':
        # Convert each coefficient to mpmath multi-precision float
        if coeffs.ndim == 1:
            return [mpmath.mpf(str(c)) for c in coeffs]
        else:
            return [[mpmath.mpf(str(c)) for c in row] for row in coeffs]
    else:
        raise ValueError(f"Unknown precision: {precision}")


def get_poles_from_a(a_coeffs, precision):
    """
    Compute poles from denominator polynomial coefficients.

    Parameters
    ----------
    a_coeffs : array-like
        Denominator coefficients a[0]=1, a[1], ..., a[N]
    precision : str
        Precision level

    Returns
    -------
    poles : ndarray (complex)
        Pole locations in the z-plane
    """
    if precision == 'mpmath':
        # Convert mpmath coefficients back to float64 for root finding
        a_float = np.array([float(c) for c in a_coeffs], dtype=np.float64)
    else:
        a_float = np.array(a_coeffs, dtype=np.float64)

    poles = np.roots(a_float)
    return poles


def get_poles_from_sos(sos_coeffs, precision):
    """
    Compute poles from SOS representation.

    Parameters
    ----------
    sos_coeffs : array-like
        SOS matrix at target precision
    precision : str
        Precision level

    Returns
    -------
    poles : ndarray (complex)
        All pole locations
    """
    if precision == 'mpmath':
        sos_float = np.array([[float(c) for c in row] for row in sos_coeffs],
                              dtype=np.float64)
    else:
        sos_float = np.array(sos_coeffs, dtype=np.float64)

    _, a_total = signal.sos2tf(sos_float)
    poles = np.roots(a_total)
    return poles


# Frequency Response
def compute_freq_response(b, a, n_points=4096, precision='float64'):
    """
    Compute the frequency response magnitude of a filter.

    Parameters
    ----------
    b, a : array-like
        Filter coefficients at target precision
    n_points : int
        Number of frequency points
    precision : str
        Precision level

    Returns
    -------
    freqs : ndarray
        Normalised frequencies (0 to 1)
    H_mag : ndarray
        Magnitude response (linear)
    """
    if precision == 'mpmath':
        b_f = np.array([float(c) for c in b], dtype=np.float64)
        a_f = np.array([float(c) for c in a], dtype=np.float64)
    else:
        b_f = np.array(b, dtype=np.float64)
        a_f = np.array(a, dtype=np.float64)

    freqs, H = signal.freqz(b_f, a_f, worN=n_points)
    freqs_norm = freqs / np.pi
    H_mag = np.abs(H)
    return freqs_norm, H_mag


if __name__ == '__main__':
    print("Testing filter_design.py...")

    # Test IIR design
    sos, b, a = design_iir_filter('butter', order=10, cutoff=0.3)
    print(f"Butterworth order-10: {len(a)-1} poles, a shape={a.shape}")

    # Test precision conversion
    a16 = convert_to_precision(a, 'float16')
    a32 = convert_to_precision(a, 'float32')
    a64 = convert_to_precision(a, 'float64')
    amp = convert_to_precision(a, 'mpmath')
    print(f"float16 a[1] = {a16[1]}")
    print(f"float32 a[1] = {a32[1]}")
    print(f"float64 a[1] = {a64[1]}")
    print(f"mpmath  a[1] = {amp[1]}")

    # Test FIR design
    h = design_fir_filter(100, cutoff=0.3)
    print(f"FIR 100-tap: {len(h)} coefficients")

    # Test poles
    poles = get_poles_from_a(a, 'float64')
    print(f"Poles: {len(poles)} poles, max |p| = {max(abs(poles)):.6f}")

    print("\nAll tests passed!")
