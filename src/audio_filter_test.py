
import numpy as np
from scipy import signal
import mpmath
import os, struct, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
mpmath.mp.dps = 50

# ── Configuration ─────────────────────────────────────────────────────────────
FS          = 48000
CUTOFF_HZ   = 8000
CUTOFF_NORM = CUTOFF_HZ / (FS / 2)
ORDER_BA    = 14
ORDER_SOS   = 18
N_IMPULSE   = 4000
N_SINE      = 6000
N_SETTLE    = 4000
N_AUDIO     = FS * 3
PRECISIONS  = ['float16', 'float32', 'float64', 'mpmath']
RESULTS_DIR = os.path.join('..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Test frequencies — multiples of fs/N_ss = 48000/2000 = 24 Hz
TEST_FREQS_HZ = [480, 1200, 2400, 4800, 7200, 8400, 12000]
COLORS = {'float16':'tab:red','float32':'tab:orange',
          'float64':'tab:blue','mpmath':'tab:green'}


# Helpers 
def convert_ba(c, p):
    if p == 'mpmath':
        return np.array([float(mpmath.mpf(str(x))) for x in c], dtype=np.float64)
    return c.astype({'float16':np.float16,'float32':np.float32,
                     'float64':np.float64}[p]).astype(np.float64)

def convert_sos(s, p):
    if p == 'mpmath':
        return np.array([[float(mpmath.mpf(str(x))) for x in r]
                         for r in s], dtype=np.float64)
    return s.astype({'float16':np.float16,'float32':np.float32,
                     'float64':np.float64}[p]).astype(np.float64)

def sm_ba(a):
    try:
        a64 = a.astype(np.float64)
        if not np.all(np.isfinite(a64)): return -999.0
        return float(1.0 - np.max(np.abs(np.roots(a64))))
    except: return -999.0

def sm_sos(sos):
    try:
        mn = np.inf
        for sec in sos:
            s64 = sec[3:].astype(np.float64)
            if not np.all(np.isfinite(s64)): return -999.0
            mn = min(mn, float(1.0 - np.max(np.abs(np.roots(s64)))))
        return mn
    except: return -999.0

def save_wav(fname, data, fs=FS):
    mx = np.max(np.abs(data))
    d  = np.clip(data / (mx + 1e-12), -1.0, 1.0)
    d16 = (d * 32767).astype(np.int16)
    with open(fname, 'wb') as f:
        n = len(d16)
        f.write(b'RIFF'); f.write(struct.pack('<I', 36+n*2))
        f.write(b'WAVEfmt '); f.write(struct.pack('<I', 16))
        f.write(struct.pack('<HHIIHH', 1, 1, fs, fs*2, 2, 16))
        f.write(b'data'); f.write(struct.pack('<I', n*2))
        f.write(d16.tobytes())

def speech_signal(n_samples, fs):
    t = np.arange(n_samples) / fs
    s = sum(np.sin(2*np.pi*150*k*t)/k for k in range(1,9) if 150*k < fs/2)
    for f in [440, 880, 2000, 4000]:
        s += 0.3 * np.sin(2*np.pi*f*t)
    return (s / np.max(np.abs(s))).astype(np.float64)


# INTERNAL OVERFLOW TRACKER
def lfilter_with_overflow_monitor(b, a, x, precision='float64'):
    """
    Manual IIR filter implementation that monitors internal state
    variables at EVERY sample step — not just the output.
    This is what Martin asked for: track WHERE INSIDE the filter
    overflow occurs.

    Direct-Form II Transposed:
      w[n] = x[n] - a1*w[n-1] - a2*w[n-2] - ... - aN*w[n-N]
      y[n] = b0*w[n] + b1*w[n-1] + ... + bM*w[n-M]

    Output Recorded:
      - max state variable value at each sample
      - first sample where overflow (inf/nan) occurs
      - which state variable overflowed first
    """
    N_samples = len(x)
    order = len(a) - 1
    w = np.zeros(order + 1)   # state variables (internal registers)

    y = np.zeros(N_samples)
    max_state_per_sample = np.zeros(N_samples)
    overflow_sample = None
    overflow_state  = None

    for n in range(N_samples):
        # State update (Direct-Form II transposed)
        w_new = float(x[n])
        for k in range(1, order + 1):
            w_new -= float(a[k]) * float(w[k-1]) if k-1 < len(w) else 0.0

        # Shift state registers
        w[1:] = w[:-1]
        w[0]  = w_new

        # Output
        y_n = sum(float(b[k]) * float(w[k]) for k in range(min(len(b), order+1)))
        y[n] = y_n

        # Monitor internal state variables
        max_state = np.max(np.abs(w))
        max_state_per_sample[n] = max_state

        # Check for first overflow
        if overflow_sample is None:
            if not np.isfinite(max_state) or (precision == 'float16' and max_state > 60000):
                overflow_sample = n
                overflow_state  = int(np.argmax(np.abs(w)))

    return y, max_state_per_sample, overflow_sample, overflow_state


def sosfilt_with_overflow_monitor(sos, x, precision='float64'):
    """
    Manual SOS filter with internal state monitoring.
    Each 2nd-order section has 2 state variables (zi).
    We monitor which section overflows first.
    """
    N_samples = len(x)
    n_sections = len(sos)

    y = x.copy().astype(np.float64)
    max_state_all = np.zeros((N_samples, n_sections))
    overflow_sample  = None
    overflow_section = None

    for sec_idx, sec in enumerate(sos):
        b_s = sec[:3].astype(np.float64)
        a_s = sec[3:].astype(np.float64)
        z = np.zeros(2)
        x_in = y.copy()
        y_out = np.zeros(N_samples)

        for n in range(N_samples):
            xn = float(x_in[n])
            yn = b_s[0]*xn + z[0]
            z[0] = b_s[1]*xn - a_s[1]*yn + z[1]
            z[1] = b_s[2]*xn - a_s[2]*yn

            y_out[n] = yn
            ms = max(abs(z[0]), abs(z[1]))
            max_state_all[n, sec_idx] = ms

            if overflow_sample is None:
                if not np.isfinite(ms) or (precision == 'float16' and ms > 60000):
                    overflow_sample  = n
                    overflow_section = sec_idx

        y = y_out

    return y, max_state_all, overflow_sample, overflow_section


# DYNAMIC RANGE CHEC
def check_dynamic_range():
    print(f"\n{'='*65}")
    print(f"DYNAMIC RANGE — Coefficient Underflow Analysis")
    print(f"float16 min representable: {np.finfo(np.float16).tiny:.2e}")
    print(f"float16 max representable: {np.finfo(np.float16).max:.2e}")
    print(f"{'='*65}")
    print(f"\n{'Order':<7}{'f16 lost':>12}{'f32 lost':>12}{'f64 lost':>12}")
    print(f"{'-'*45}")
    for order in range(2, 23, 2):
        sos = signal.butter(order, CUTOFF_NORM, output='sos').astype(np.float64)
        row = f"{order:<7}"
        for p, dt in [('f16',np.float16),('f32',np.float32),('f64',np.float64)]:
            sq = sos.astype(dt)
            lost = int(np.sum(sq[sos != 0] == 0))
            total = int(np.sum(sos != 0))
            flag = ' *' if lost > 0 else ''
            row += f"{lost}/{total}{flag:>5}"
        print(row)
    print("* = coefficients lost to underflow at that precision")
    print(f"\nSOS order {ORDER_SOS} chosen: highest with no float16 underflow")


# TEST 1: IMPULSE RESPONSE
def test1_impulse():
    print(f"\n{'='*65}")
    print(f"TEST 1: IMPULSE RESPONSE METHOD")
    print(f"Feed delta[n] -> filter -> DTFT of h[n] -> compare vs freqz()")
    print(f"{'='*65}")

    b_ref,a_ref = signal.butter(ORDER_BA, CUTOFF_NORM, output='ba')
    sos_ref     = signal.butter(ORDER_SOS, CUTOFF_NORM, output='sos')
    b_ref=b_ref.astype(np.float64); a_ref=a_ref.astype(np.float64)
    sos_ref=sos_ref.astype(np.float64)

    x = np.zeros(N_IMPULSE); x[0] = 1.0
    freqs = np.linspace(0, FS/2, N_IMPULSE//2)
    n_arr = np.arange(N_IMPULSE)
    w_ref, H_ref = signal.freqz(b_ref, a_ref, worN=N_IMPULSE//2, fs=FS)
    H_ref_mag = np.abs(H_ref)

    dtft_results = {}

    for label, sid, order in [('BA  (Direct-Form)','ba', ORDER_BA),
                               ('SOS (Ladder)',    'sos',ORDER_SOS)]:
        print(f"\n  {label} order {order}")
        print(f"  {'Precision':<10}{'Max|h[n]|':>12}{'DTFT_err':>12}"
              f"{'Max_state':>12}{'1st_OVF_sample':>16}{'Status':>8}")
        print(f"  {'-'*72}")

        dtft_results[sid] = {}
        for p in PRECISIONS:
            if sid == 'ba':
                b_q=convert_ba(b_ref,p); a_q=convert_ba(a_ref,p)
                h, ms_arr, ovf_samp, ovf_st = \
                    lfilter_with_overflow_monitor(b_q, a_q, x, p)
                stab = sm_ba(a_q)
            else:
                s_q=convert_sos(sos_ref,p)
                h, ms_arr, ovf_samp, ovf_sec = \
                    sosfilt_with_overflow_monitor(s_q, x, p)
                stab = sm_sos(s_q)

            h = np.array(h, dtype=np.float64)
            mx_h = float(np.max(np.abs(h[np.isfinite(h)]))) \
                   if np.any(np.isfinite(h)) else np.inf
            mx_state = float(np.max(ms_arr[np.isfinite(ms_arr)])) \
                       if np.any(np.isfinite(ms_arr)) else np.inf

            if not np.any(np.isfinite(h)):
                print(f"  {p:<10}{'OVERFLOW':>12}{'---':>12}"
                      f"{mx_state:>12.2e}{str(ovf_samp):>16}{'FAIL':>8}")
                dtft_results[sid][p] = None
                continue

            H_dtft = np.array([np.sum(h*np.exp(-1j*2*np.pi*f/FS*n_arr))
                                for f in freqs])
            err = float(np.max(np.abs(np.abs(H_dtft)-H_ref_mag)))
            status = 'OK' if stab > 0 and err < 2.0 else 'DEGRADE'
            ovf_str = str(ovf_samp) if ovf_samp else 'None'

            print(f"  {p:<10}{mx_h:>12.4f}{err:>12.2e}"
                  f"{mx_state:>12.4f}{ovf_str:>16}{status:>8}")
            dtft_results[sid][p] = (freqs, np.abs(H_dtft))

    return dtft_results, freqs, H_ref_mag, w_ref


# TEST 2: SINUSOIDAL METHOD
def test2_sinusoidal():
    print(f"\n{'='*65}")
    print(f"TEST 2: SINUSOIDAL METHOD (Standard Test Tone)")
    print(f"Feed sin(2pi*f*n) -> filter -> measure steady-state |H|")
    print(f"Monitor INTERNAL state variables for overflow")
    print(f"Save filtered sine waves as .wav for listening")
    print(f"{'='*65}")

    b_ref,a_ref = signal.butter(ORDER_BA, CUTOFF_NORM, output='ba')
    sos_ref     = signal.butter(ORDER_SOS, CUTOFF_NORM, output='sos')
    b_ref=b_ref.astype(np.float64); a_ref=a_ref.astype(np.float64)
    sos_ref=sos_ref.astype(np.float64)

    sine_results = {}

    for label, sid, order in [('BA  (Direct-Form)','ba', ORDER_BA),
                               ('SOS (Ladder)',    'sos',ORDER_SOS)]:
        print(f"\n  {label} order {order}")
        print(f"  {'Freq(Hz)':<10}{'|H|_theory':>11}{'Precision':<10}"
              f"{'|H|_meas':>10}{'err':>10}{'Max_state':>12}"
              f"{'1st_OVF':>10}{'Overflow':>10}")
        print(f"  {'-'*85}")

        sine_results[sid] = {}
        for f_hz in TEST_FREQS_HZ:
            _, H_th = signal.freqz(b_ref, a_ref, worN=[f_hz], fs=FS)
            th = float(np.abs(H_th[0]))
            n  = np.arange(N_SINE)
            x  = np.sin(2*np.pi*f_hz/FS*n)

            for p in PRECISIONS:
                if sid == 'ba':
                    b_q=convert_ba(b_ref,p); a_q=convert_ba(a_ref,p)
                    y, ms_arr, ovf_s, _ = \
                        lfilter_with_overflow_monitor(b_q, a_q, x, p)
                else:
                    s_q=convert_sos(sos_ref,p)
                    y, ms_arr, ovf_s, ovf_sec = \
                        sosfilt_with_overflow_monitor(s_q, x, p)

                y=np.array(y,dtype=np.float64)
                mx_state = float(np.max(ms_arr[np.isfinite(ms_arr)])) \
                           if np.any(np.isfinite(ms_arr)) else np.inf
                ovf = not np.all(np.isfinite(y)) or \
                      (p=='float16' and mx_state > 60000)
                ovf_str = str(ovf_s) if ovf_s else 'None'

                y_ss = y[N_SETTLE:]; x_ss = x[N_SETTLE:]
                if not np.any(np.isfinite(y_ss)) or ovf:
                    amp=np.nan; err=np.nan
                    print(f"  {f_hz:<10}{th:>11.4f}{p:<10}"
                          f"{'OVERFLOW':>10}{'---':>10}"
                          f"{mx_state:>12.2e}{ovf_str:>10}{'YES':>10}")
                else:
                    amp = np.std(y_ss)/(np.std(x_ss)+1e-12)
                    err = abs(amp-th)
                    print(f"  {f_hz:<10}{th:>11.4f}{p:<10}"
                          f"{amp:>10.4f}{err:>10.2e}"
                          f"{mx_state:>12.4f}{ovf_str:>10}"
                          f"{'YES' if ovf else 'NO':>10}")

                # Save filtered sine wave as .wav for listening
                fname = os.path.join(RESULTS_DIR,
                    f'sine_{f_hz}Hz_{sid.upper()}_order{order}_{p}.wav')
                y_save = y if np.any(np.isfinite(y)) else np.zeros_like(x)
                save_wav(fname, y_save)

                key = (f_hz, p)
                sine_results[sid][key] = {'theory':th,'measured':amp,'err':err}
            print()

    return sine_results


# TEST 3: AUDIO SIGNAL
def test3_audio():
    print(f"\n{'='*65}")
    print(f"TEST 3: AUDIO SIGNAL TEST (Subjective Listening)")
    print(f"Speech-like signal -> filter -> save .wav -> listen")
    print(f"Measure SNR = 10*log10(signal_power / noise_power)")
    print(f"{'='*65}")

    x = speech_signal(N_AUDIO, FS)
    save_wav(os.path.join(RESULTS_DIR,'audio_input.wav'), x)
    print(f"\n  Saved: audio_input.wav")
    print(f"  Content: 150Hz speech + harmonics + 440/880/2000/4000Hz tones")

    b_ref,a_ref = signal.butter(ORDER_BA,  CUTOFF_NORM, output='ba')
    sos_ref     = signal.butter(ORDER_SOS, CUTOFF_NORM, output='sos')
    b_ref=b_ref.astype(np.float64); a_ref=a_ref.astype(np.float64)
    sos_ref=sos_ref.astype(np.float64)

    y_ref_ba  = signal.lfilter(b_ref, a_ref, x)
    y_ref_sos = signal.sosfilt(sos_ref, x)
    save_wav(os.path.join(RESULTS_DIR,'audio_BA_float64_ref.wav'),  y_ref_ba)
    save_wav(os.path.join(RESULTS_DIR,'audio_SOS_float64_ref.wav'), y_ref_sos)

    print(f"\n  {'Structure':<18}{'Precision':<10}{'Max output':>12}"
          f"{'Max_state':>12}{'1st_OVF':>10}{'SNR(dB)':>10}  File")
    print(f"  {'-'*90}")

    for label, sid, order, ref in [
            ('BA  (Direct-Form)','ba', ORDER_BA,  y_ref_ba),
            ('SOS (Ladder)',     'sos',ORDER_SOS, y_ref_sos)]:
        for p in PRECISIONS:
            if sid == 'ba':
                b_q=convert_ba(b_ref,p); a_q=convert_ba(a_ref,p)
                y, ms_arr, ovf_s, _ = \
                    lfilter_with_overflow_monitor(b_q, a_q, x, p)
            else:
                s_q=convert_sos(sos_ref,p)
                y, ms_arr, ovf_s, _ = \
                    sosfilt_with_overflow_monitor(s_q, x, p)

            y=np.array(y,dtype=np.float64)
            mx_out = float(np.max(np.abs(y[np.isfinite(y)]))) \
                     if np.any(np.isfinite(y)) else np.inf
            mx_state = float(np.max(ms_arr[np.isfinite(ms_arr)])) \
                       if np.any(np.isfinite(ms_arr)) else np.inf
            ovf_str = str(ovf_s) if ovf_s else 'None'

            if not np.any(np.isfinite(y)):
                snr_str='OVERFLOW'; y_sv=np.zeros_like(x)
            else:
                noise=y-ref
                sp=np.mean(ref**2); np_=np.mean(noise**2)
                snr_db=10*np.log10(sp/np_) if np_>0 else np.inf
                snr_str=f"{snr_db:.1f}" if np.isfinite(snr_db) else "inf"
                y_sv=y

            fname=f"audio_{sid.upper()}_order{order}_{p}.wav"
            save_wav(os.path.join(RESULTS_DIR,fname), y_sv)
            print(f"  {label:<18}{p:<10}{mx_out:>12.4f}"
                  f"{mx_state:>12.4f}{ovf_str:>10}{snr_str:>10}  {fname}")
        print()


# TEST 4: FREQUENCY RESPONSE COMPARISON PLOT
def test4_freqz_comparison_plot(dtft_results, freqs_dtft, H_ref_mag, w_ref):
    """
    Plot all three methods on the same axes:
    1. freqz() theoretical reference
    2. Impulse response DTFT
    3. Sinusoidal method measured points

    """
    print(f"\n{'='*65}")
    print(f"TEST 4: FREQUENCY RESPONSE COMPARISON PLOT")
    print(f"freqz() vs Impulse DTFT vs Sinusoidal method — same axes")
    print(f"{'='*65}")

    b_ref,a_ref = signal.butter(ORDER_BA, CUTOFF_NORM, output='ba')
    sos_ref     = signal.butter(ORDER_SOS, CUTOFF_NORM, output='sos')
    b_ref=b_ref.astype(np.float64); a_ref=a_ref.astype(np.float64)
    sos_ref=sos_ref.astype(np.float64)

    for sid, order, label in [('ba', ORDER_BA,  'Direct-Form (BA)'),
                               ('sos',ORDER_SOS, 'Ladder (SOS)')]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Frequency Response Comparison — {label} order {order}\n'
                     f'freqz() vs Impulse DTFT vs Sinusoidal Method\n'
                     f'fs={FS}Hz  cutoff={CUTOFF_HZ}Hz',
                     fontsize=11, fontweight='bold')

        for idx, p in enumerate(PRECISIONS):
            ax = axes[idx//2][idx%2]

            # 1 — freqz() theoretical reference
            if sid == 'ba':
                b_q=convert_ba(b_ref,p); a_q=convert_ba(a_ref,p)
                w_q, H_q = signal.freqz(b_q, a_q, worN=1024, fs=FS)
            else:
                s_q=convert_sos(sos_ref,p)
                w_q, H_q = signal.sosfreqz(s_q, worN=1024, fs=FS)

            ax.plot(w_q, 20*np.log10(np.abs(H_q)+1e-300),
                    '-', color='tab:blue', lw=1.5, label='freqz() theoretical', zorder=3)

            # 2 — Impulse response DTFT
            if dtft_results.get(sid,{}).get(p) is not None:
                f_dtft, H_dtft_mag = dtft_results[sid][p]
                ax.plot(f_dtft, 20*np.log10(H_dtft_mag+1e-300),
                        '--', color='tab:orange', lw=1.5,
                        label='Impulse DTFT', zorder=2)

            # 3 — Sinusoidal method measured points
            x_pts = []; y_pts = []
            for f_hz in TEST_FREQS_HZ:
                _, H_th = signal.freqz(b_ref, a_ref, worN=[f_hz], fs=FS)
                n = np.arange(N_SINE)
                x_in = np.sin(2*np.pi*f_hz/FS*n)
                if sid == 'ba':
                    b_q2=convert_ba(b_ref,p); a_q2=convert_ba(a_ref,p)
                    y_out = signal.lfilter(b_q2, a_q2, x_in)
                else:
                    s_q2=convert_sos(sos_ref,p)
                    y_out = signal.sosfilt(s_q2, x_in)
                y_ss = np.array(y_out[N_SETTLE:], dtype=np.float64)
                x_ss = x_in[N_SETTLE:]
                if np.any(np.isfinite(y_ss)):
                    amp = np.std(y_ss)/(np.std(x_ss)+1e-12)
                    if np.isfinite(amp) and amp > 0:
                        x_pts.append(f_hz)
                        y_pts.append(20*np.log10(amp+1e-300))

            if x_pts:
                ax.scatter(x_pts, y_pts, color='tab:red', s=60, zorder=4,
                           label='Sinusoidal measured', marker='o')

            # Reference (float64) in grey
            ax.plot(w_ref, 20*np.log10(H_ref_mag+1e-300),
                    ':', color='gray', lw=1, label='float64 ref', zorder=1)

            ax.axvline(CUTOFF_HZ, color='black', ls=':', lw=0.8)
            ax.set_ylim(-80, 10)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('|H(f)| (dB)')
            ax.set_title(f'{p}')
            ax.legend(fontsize=7, loc='lower left')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = os.path.join(RESULTS_DIR,
                f'freqz_comparison_{sid}_order{order}.png')
        plt.savefig(fname, dpi=120)
        plt.close()
        print(f"  Saved: {fname}")


# DETERIORATION SWEEP
def deterioration_sweep():
    print(f"\n{'='*65}")
    print(f"DETERIORATION SWEEP")
    print(f"At what filter order does SNR drop below 40 dB?")
    print(f"{'='*65}")

    x = speech_signal(FS*2, FS)
    print(f"\n{'Order':<6}  "
          f"{'BA_f16':>9}{'BA_f32':>9}{'BA_f64':>9}  |  "
          f"{'SOS_f16':>9}{'SOS_f32':>9}{'SOS_f64':>9}")
    print("-"*70)

    for order in range(2, 23, 2):
        b,a=signal.butter(order,CUTOFF_NORM,output='ba')
        sos=signal.butter(order,CUTOFF_NORM,output='sos')
        b=b.astype(np.float64); a=a.astype(np.float64)
        sos=sos.astype(np.float64)
        ref_ba  = signal.lfilter(b,a,x)
        ref_sos = signal.sosfilt(sos,x)

        row = f"{order:<6}  "
        for sid, ref in [('ba',ref_ba),('sos',ref_sos)]:
            if sid=='sos': row+="  |  "
            for dt in [np.float16,np.float32,np.float64]:
                try:
                    if sid=='ba':
                        y=signal.lfilter(b.astype(dt).astype(np.float64),
                                         a.astype(dt).astype(np.float64),x)
                    else:
                        y=signal.sosfilt(sos.astype(dt).astype(np.float64),x)
                    y=np.array(y,dtype=np.float64)
                    if not np.any(np.isfinite(y)):
                        row+=f"{'OVF':>9}"
                    else:
                        noise=y-ref; sp=np.mean(ref**2); np_=np.mean(noise**2)
                        snr=10*np.log10(sp/np_) if np_>0 else 999.0
                        flag='*' if snr<40 else ' '
                        row+=f"{snr:>8.1f}{flag}"
                except: row+=f"{'ERR':>9}"
        print(row)

    print("\n* = SNR below 40 dB (audibly deteriorated)")


# MAIN
if __name__ == '__main__':
    print("COMPLETE AUDIO FILTER TEST — ALL METHODS")
    print("Student: Darshan Mahadeva Naika | A00090581 | EEN1095")
    print(f"fs={FS}Hz | cutoff={CUTOFF_HZ}Hz")
    print(f"BA order={ORDER_BA} | SOS order={ORDER_SOS}")

    check_dynamic_range()
    dtft_res, freqs_d, H_ref_mag, w_ref = test1_impulse()
    test2_sinusoidal()
    test3_audio()
    test4_freqz_comparison_plot(dtft_res, freqs_d, H_ref_mag, w_ref)
    deterioration_sweep()

    print(f"\n{'='*65}")
    print(f"ALL FILES SAVED TO: {os.path.abspath(RESULTS_DIR)}/")
    print(f"\nLISTENING GUIDE:")
    print(f"  Step 1: audio_input.wav              <- original signal")
    print(f"  Step 2: audio_BA_float64_ref.wav     <- BA reference (clean)")
    print(f"  Step 3: audio_BA_order{ORDER_BA}_float16.wav <- BA float16 (distorted?)")
    print(f"  Step 4: audio_SOS_float64_ref.wav    <- SOS reference (clean)")
    print(f"  Step 5: audio_SOS_order{ORDER_SOS}_float16.wav <- SOS float16 (better?)")
    print(f"\n  KEY: compare Step 3 vs Step 5 — same precision, different structure")
    print(f"\nSINE WAVE LISTENING GUIDE:")
    print(f"  sine_4800Hz_BA_order{ORDER_BA}_float16.wav  <- BA at 4800Hz float16")
    print(f"  sine_4800Hz_SOS_order{ORDER_SOS}_float16.wav <- SOS at 4800Hz float16")
    print(f"  These should sound most different — 4800Hz showed largest error")
    print(f"\nPLOTS:")
    print(f"  freqz_comparison_ba_order{ORDER_BA}.png  <- all 3 methods on one graph (BA)")
    print(f"  freqz_comparison_sos_order{ORDER_SOS}.png <- all 3 methods on one graph (SOS)")
