"""
Validate filter implementation using two independent test methods:

1. Impulse Response Method:
   Feed a digital impulse through the filter, obtain h[n], compute its
   frequency response, and compare to scipy.signal.freqz().

2. Sine Wave Method:
   Feed a steady-state sine wave at each test frequency through the
   filter (after discarding the transient). Measure output amplitude
   ratio, phase shift, and Total Harmonic Distortion (THD).

N_SETTLE is chosen so the post-transient window contains an integer
number of cycles at every test frequency (avoids spectral leakage).
"""

import numpy as np
from scipy import signal
import mpmath
import warnings
warnings.filterwarnings('ignore')

mpmath.mp.dps = 50

PRECISIONS = ['float16', 'float32', 'float64', 'mpmath']

TEST_FREQS = {
    'Passband-low':  0.05,
    'Passband-mid':  0.15,
    'Near-cutoff':   0.28,
    'Stopband-near': 0.35,
    'Stopband-far':  0.45,
}

N_IMPULSE = 2000
N_SINE    = 2000
N_SETTLE  = 1000   # N_ss = 1000 -> integer cycles for all TEST_FREQS (avoids leakage)


def design_filter(ftype='butter', order=10, cutoff=0.3):
    if ftype == 'butter':
        b, a = signal.butter(order, cutoff, btype='low', output='ba')
    elif ftype == 'ellip':
        b, a = signal.ellip(order, 1.0, 40.0, cutoff, btype='low', output='ba')
    return b.astype(np.float64), a.astype(np.float64)


def convert(coeffs, precision):
    if precision == 'float16':
        return coeffs.astype(np.float16)
    elif precision == 'float32':
        return coeffs.astype(np.float32)
    elif precision == 'float64':
        return coeffs.astype(np.float64)
    elif precision == 'mpmath':
        return np.array([float(mpmath.mpf(str(c))) for c in coeffs], dtype=np.float64)


def impulse_response_freq(b, a, freq, n_samples=N_IMPULSE):
    x = np.zeros(n_samples)
    x[0] = 1.0
    h = signal.lfilter(b, a, x)
    w = freq * np.pi
    n = np.arange(n_samples)
    H = np.sum(h * np.exp(-1j * w * n))
    return np.abs(H), np.angle(H), h


def sine_wave_response(b, a, freq, n_samples=N_SINE, n_settle=N_SETTLE):
    n = np.arange(n_samples)
    w = freq * np.pi
    x = np.sin(w * n)
    y = signal.lfilter(b, a, x)

    x_ss = x[n_settle:]
    y_ss = y[n_settle:]

    amp_ratio = np.std(y_ss) / np.std(x_ss)

    N = len(y_ss)
    Y = np.fft.fft(y_ss)
    X = np.fft.fft(x_ss)
    freqs_fft = np.fft.fftfreq(N, d=1.0)
    target_cycle = w / (2 * np.pi)
    idx = np.argmin(np.abs(freqs_fft - target_cycle))

    phase_shift = np.angle(Y[idx]) - np.angle(X[idx])
    phase_shift = np.mod(phase_shift + np.pi, 2 * np.pi) - np.pi

    return amp_ratio, phase_shift, y_ss


def freqz_response(b, a, freq):
    w, H = signal.freqz(b, a, worN=[freq * np.pi])
    return np.abs(H[0]), np.angle(H[0])


def total_harmonic_distortion(y_ss, freq, n_harmonics=5):
    N = len(y_ss)
    Y = np.abs(np.fft.fft(y_ss))
    freqs_fft = np.fft.fftfreq(N, d=1.0)
    target_cycle = (freq * np.pi) / (2 * np.pi)

    fund_idx = np.argmin(np.abs(freqs_fft - target_cycle))
    fund_amp = Y[fund_idx]
    if fund_amp < 1e-12:
        return np.nan

    harmonic_energy = 0.0
    for k in range(2, n_harmonics + 1):
        harm_cycle = k * target_cycle
        if harm_cycle >= 0.5:
            break
        harm_idx = np.argmin(np.abs(freqs_fft - harm_cycle))
        harmonic_energy += Y[harm_idx] ** 2

    return float(np.sqrt(harmonic_energy) / fund_amp)


def run_validation(ftype='butter', order=10):
    print(f"\n=== {ftype.upper()} ORDER {order} ===")
    print(f"{'Freq':<14}{'Prec':<9}{'|H|':>10}{'Phase':>9}{'|H|err':>11}{'Pherr':>10}{'THD':>11}{'Match':>11}")

    b_ref, a_ref = design_filter(ftype, order)

    for label, freq in TEST_FREQS.items():
        H_mag_ref, H_phase_ref = freqz_response(b_ref, a_ref, freq)

        for precision in PRECISIONS:
            b_q = convert(b_ref, precision)
            a_q = convert(a_ref, precision)

            H_mag_imp, H_phase_imp, _ = impulse_response_freq(b_q, a_q, freq)
            amp_ratio, phase_shift, y_ss = sine_wave_response(b_q, a_q, freq)
            thd = total_harmonic_distortion(y_ss, freq)

            imp_err   = abs(H_mag_imp - H_mag_ref)
            ph_imp_err = abs(np.degrees(H_phase_imp) - np.degrees(H_phase_ref))
            method_match = abs(H_mag_imp - amp_ratio)  # impulse vs sine, should be ~0

            if np.isnan(H_mag_imp):
                print(f"{label:<14}{precision:<9}{'UNSTABLE (nan)':>52}")
            else:
                print(f"{label:<14}{precision:<9}{H_mag_imp:>10.6f}{np.degrees(H_phase_imp):>9.2f}"
                      f"{imp_err:>11.2e}{ph_imp_err:>10.2e}{thd:>11.2e}{method_match:>11.2e}")


run_validation('butter', order=10)
run_validation('butter', order=20)

print("\nNotes:")
print("- |H|err / Pherr = deviation from freqz() reference (float64).")
print("- THD ~1e-15 means the filter behaves linearly (no harmonic distortion).")
print("- Match = |H|(impulse) - |H|(sine); near-zero confirms both methods agree.")