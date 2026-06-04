"""
Performance metrics for the multi-precision digital filter evaluation framework.
Computes pole displacement, stability margin, sensitivity norm,
frequency response deviation, and roundoff noise accumulation.
"""

import numpy as np
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

N_FREQ_POINTS  = 4096    # frequency grid size
N_TRIALS       = 10      # independent perturbation trials
PERTURB_AMP    = 1e-6    # perturbation amplitude (relative to coeff magnitude)
N_SAMPLES      = 1000    # output samples for roundoff noise measurement


# ── IIR Metrics ───────────────────────────────────────────────────────────────
def pole_displacement(poles_ref, poles_quant):
    """
    Compute pole displacement: L2 norm difference between
    reference poles and quantised poles.

    Δp = ||p_orig - p_quant||_2

    Parameters
    ----------
    poles_ref : ndarray (complex)
        Reference poles (float64)
    poles_quant : ndarray (complex)
        Quantised poles at target precision

    Returns
    -------
    displacement : float
        L2 norm of pole displacement
    """
    # Sort poles by angle for consistent pairing
    ref_sorted   = poles_ref[np.argsort(np.angle(poles_ref))]
    quant_sorted = poles_quant[np.argsort(np.angle(poles_quant))]

    # Trim to same length if needed
    n = min(len(ref_sorted), len(quant_sorted))
    displacement = np.linalg.norm(ref_sorted[:n] - quant_sorted[:n])
    return float(displacement)


def stability_margin(poles):
    """
    Compute stability margin: minimum distance of any pole from the unit circle.

    SM = min_k { 1 - |p_k| }

    A positive value means all poles inside unit circle (stable).
    A negative value means at least one pole outside (unstable).

    Parameters
    ----------
    poles : ndarray (complex)
        Pole locations

    Returns
    -------
    margin : float
        Minimum stability margin (negative = unstable)
    is_stable : bool
        True if all poles inside unit circle
    """
    magnitudes = np.abs(poles)
    margin = float(1.0 - np.max(magnitudes))
    is_stable = bool(np.all(magnitudes < 1.0))
    return margin, is_stable


def sensitivity_norm(b_ref, a_ref, b_quant, a_quant,
                     n_trials=N_TRIALS, perturb_amp=PERTURB_AMP,
                     n_points=N_FREQ_POINTS):
    """
    Compute sensitivity norm: average maximum frequency response deviation
    under small coefficient perturbations.

    S = ||H_perturbed - H_reference||_inf  (averaged over n_trials)

    Parameters
    ----------
    b_ref, a_ref : ndarray
        Reference coefficients (float64)
    b_quant, a_quant : ndarray
        Quantised coefficients at target precision (cast to float64 for freqz)
    n_trials : int
        Number of independent perturbation trials
    perturb_amp : float
        Perturbation amplitude relative to coefficient magnitude
    n_points : int
        Frequency grid size

    Returns
    -------
    s_norm : float
        Average L-infinity sensitivity norm across trials
    """
    b_r = np.array(b_ref, dtype=np.float64)
    a_r = np.array(a_ref, dtype=np.float64)
    b_q = np.array(b_quant, dtype=np.float64)
    a_q = np.array(a_quant, dtype=np.float64)

    # Reference frequency response
    _, H_ref = signal.freqz(b_r, a_r, worN=n_points)
    H_ref_mag = np.abs(H_ref)

    norms = []
    for _ in range(n_trials):
        # Add Gaussian perturbation to quantised coefficients
        b_pert = b_q + np.random.normal(0, perturb_amp * np.abs(b_q).mean(), b_q.shape)
        a_pert = a_q + np.random.normal(0, perturb_amp * np.abs(a_q).mean(), a_q.shape)
        a_pert[0] = 1.0  # ensure a[0] = 1

        try:
            _, H_pert = signal.freqz(b_pert, a_pert, worN=n_points)
            H_pert_mag = np.abs(H_pert)
            norm = np.max(np.abs(H_pert_mag - H_ref_mag))
            norms.append(norm)
        except Exception:
            norms.append(np.nan)

    valid = [n for n in norms if not np.isnan(n)]
    return float(np.mean(valid)) if valid else np.nan


# ── FIR Metrics ───────────────────────────────────────────────────────────────

def freq_response_deviation(h_ref, h_quant, n_points=N_FREQ_POINTS):
    """
    Compute frequency response deviation for FIR filters.

    Dev = ||H_precision - H_reference||_inf

    Parameters
    ----------
    h_ref : ndarray
        Reference FIR coefficients (float64)
    h_quant : ndarray
        Quantised FIR coefficients at target precision

    Returns
    -------
    deviation : float
        L-infinity frequency response deviation
    deviation_db : float
        Deviation in dB
    """
    h_r = np.array(h_ref,   dtype=np.float64)
    h_q = np.array(h_quant, dtype=np.float64)

    _, H_ref  = signal.freqz(h_r, worN=n_points)
    _, H_quant = signal.freqz(h_q, worN=n_points)

    H_ref_mag   = np.abs(H_ref)
    H_quant_mag = np.abs(H_quant)

    deviation = float(np.max(np.abs(H_quant_mag - H_ref_mag)))

    # Convert to dB (avoid log(0))
    ref_max = np.max(H_ref_mag)
    if ref_max > 0 and deviation > 0:
        deviation_db = 20 * np.log10(deviation / ref_max + 1e-12)
    else:
        deviation_db = -np.inf

    return deviation, deviation_db


def roundoff_noise(b_ref, a_ref, b_quant, a_quant,
                   n_samples=N_SAMPLES, is_fir=False):
    """
    Compute roundoff noise accumulation over n_samples.

    Noise = ||y_precision - y_reference||_2

    Parameters
    ----------
    b_ref, a_ref : ndarray
        Reference filter coefficients (float64)
    b_quant, a_quant : ndarray
        Quantised coefficients at target precision
    n_samples : int
        Number of output samples to filter
    is_fir : bool
        If True, a = [1] (FIR filter)

    Returns
    -------
    noise : float
        L2 norm of accumulated roundoff error
    """
    np.random.seed(42)   # reproducible input
    x = np.random.randn(n_samples)

    b_r = np.array(b_ref,   dtype=np.float64)
    a_r = np.array(a_ref,   dtype=np.float64)
    b_q = np.array(b_quant, dtype=np.float64)
    a_q = np.array(a_quant, dtype=np.float64)

    try:
        y_ref   = signal.lfilter(b_r, a_r, x)
        y_quant = signal.lfilter(b_q, a_q, x)
        noise = float(np.linalg.norm(y_quant - y_ref))
    except Exception:
        noise = np.nan

    return noise


# ── Combined Metric Computation ───────────────────────────────────────────────

def compute_iir_metrics(b_ref, a_ref, b_quant, a_quant,
                         poles_ref, poles_quant, precision_name):
    """
    Compute all five IIR metrics for one configuration.

    Returns
    -------
    dict with keys:
        pole_displacement, stability_margin, is_stable,
        sensitivity_norm, roundoff_noise, precision
    """
    pd   = pole_displacement(poles_ref, poles_quant)
    sm, stable = stability_margin(poles_quant)
    sn   = sensitivity_norm(b_ref, a_ref, b_quant, a_quant)
    rn   = roundoff_noise(b_ref, a_ref, b_quant, a_quant)

    return {
        'precision':         precision_name,
        'pole_displacement': pd,
        'stability_margin':  sm,
        'is_stable':         stable,
        'sensitivity_norm':  sn,
        'roundoff_noise':    rn,
    }


def compute_fir_metrics(h_ref, h_quant, precision_name):
    """
    Compute FIR metrics for one configuration.

    Returns
    -------
    dict with keys:
        freq_deviation, freq_deviation_db, roundoff_noise, precision
    """
    dev, dev_db = freq_response_deviation(h_ref, h_quant)
    a_ones = np.array([1.0])
    rn = roundoff_noise(h_ref, a_ones, h_quant, a_ones, is_fir=True)

    return {
        'precision':           precision_name,
        'freq_deviation':      dev,
        'freq_deviation_db':   dev_db,
        'roundoff_noise':      rn,
    }


if __name__ == '__main__':
    print("Testing metrics.py...")
    from filter_design import design_iir_filter, convert_to_precision, get_poles_from_a

    # Design a test filter
    _, b, a = design_iir_filter('butter', order=10, cutoff=0.3)
    poles_ref = get_poles_from_a(a, 'float64')

    print(f"Reference poles (first 3): {poles_ref[:3]}")
    print(f"Stability margin (ref): {stability_margin(poles_ref)}")

    # Test at float32
    b32 = convert_to_precision(b, 'float32')
    a32 = convert_to_precision(a, 'float32')
    poles32 = get_poles_from_a(a32, 'float32')

    pd = pole_displacement(poles_ref, poles32)
    sm, stable = stability_margin(poles32)
    sn = sensitivity_norm(b, a, b32, a32, n_trials=3)
    rn = roundoff_noise(b, a, b32, a32)

    print(f"\nFloat32 results for Butterworth order-10:")
    print(f"  Pole displacement:  {pd:.2e}")
    print(f"  Stability margin:   {sm:.6f} (stable={stable})")
    print(f"  Sensitivity norm:   {sn:.2e}")
    print(f"  Roundoff noise:     {rn:.2e}")

    print("\nAll metrics tests passed!")
