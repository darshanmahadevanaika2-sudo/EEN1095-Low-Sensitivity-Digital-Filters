
import numpy as np
from scipy import signal
import mpmath
import os, struct, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
mpmath.mp.dps = 50

# AUDIO Configuration
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

TEST_FREQS_HZ = [480, 1200, 2400, 4800, 7200, 8400, 12000]
COLORS = {'float16':'tab:red','float32':'tab:orange',
          'float64':'tab:blue','mpmath':'tab:green'}

# VLF Configuration
FS_VLF          = 1000
CUTOFF_VLF      = 100
CUTOFF_NORM_VLF = CUTOFF_VLF / (FS_VLF / 2)
ORDER_BA_VLF    = 8    # Highest stable at float16 for VLF (SM=+0.045)
ORDER_SOS_VLF   = 12   # Highest with no float16 coefficient underflow
N_IMPULSE_VLF   = 2000
N_SINE_VLF      = 4000
N_SETTLE_VLF    = 2000
N_AUDIO_VLF     = FS_VLF * 5
TEST_FREQS_VLF  = [10, 50, 100, 150, 200, 300, 400]


# Shared Helpers
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
    """Speech-like test signal: 150Hz voiced fundamental + harmonics + test tones."""
    t = np.arange(n_samples) / fs
    s = sum(np.sin(2*np.pi*150*k*t)/k for k in range(1,9) if 150*k < fs/2)
    for f in [440, 880, 2000, 4000]:
        s += 0.3 * np.sin(2*np.pi*f*t)
    return (s / np.max(np.abs(s))).astype(np.float64)

def vlf_signal(n_samples, fs):
    """VLF test signal: 50Hz power line + harmonics + geophysical components."""
    t = np.arange(n_samples) / fs
    s  = np.sin(2*np.pi*50*t)
    s += 0.5 * np.sin(2*np.pi*100*t)
    s += 0.3 * np.sin(2*np.pi*150*t)
    s += 0.2 * np.sin(2*np.pi*200*t)
    s += 0.4 * np.sin(2*np.pi*10*t)
    s += 0.3 * np.sin(2*np.pi*30*t)
    s += 0.1 * np.sin(2*np.pi*1*t)
    return (s / np.max(np.abs(s))).astype(np.float64)


# Internal Overflow Monitor
def lfilter_with_overflow_monitor(b, a, x, precision='float64'):
    """Monitor internal state variables at EVERY sample - track WHERE overflow occurs."""
    N_samples = len(x)
    order = len(a) - 1
    w = np.zeros(order + 1)
    y = np.zeros(N_samples)
    max_state_per_sample = np.zeros(N_samples)
    overflow_sample = None
    overflow_state  = None

    for n in range(N_samples):
        w_new = float(x[n])
        for k in range(1, order + 1):
            w_new -= float(a[k]) * float(w[k-1]) if k-1 < len(w) else 0.0
        w[1:] = w[:-1]
        w[0]  = w_new
        y_n = sum(float(b[k]) * float(w[k]) for k in range(min(len(b), order+1)))
        y[n] = y_n
        max_state = np.max(np.abs(w))
        max_state_per_sample[n] = max_state
        if overflow_sample is None:
            if not np.isfinite(max_state) or (precision == 'float16' and max_state > 60000):
                overflow_sample = n
                overflow_state  = int(np.argmax(np.abs(w)))

    return y, max_state_per_sample, overflow_sample, overflow_state


def sosfilt_with_overflow_monitor(sos, x, precision='float64'):
    """Monitor SOS internal state variables - track which section overflows first."""
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

# PART 1 - AUDIO RANGE TESTS (fs=48000Hz, cutoff=8000Hz)

def check_dynamic_range():
    print(f"\n{'='*65}")
    print(f"DYNAMIC RANGE - Coefficient Underflow Analysis (AUDIO)")
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


def test1_impulse():
    print(f"\n{'='*65}")
    print(f"TEST 1: IMPULSE RESPONSE METHOD (AUDIO)")
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
              f"{'Max_state':>12}{'1st_OVF':>16}{'Status':>8}")
        print(f"  {'-'*72}")

        dtft_results[sid] = {}
        for p in PRECISIONS:
            if sid == 'ba':
                b_q=convert_ba(b_ref,p); a_q=convert_ba(a_ref,p)
                h,ms_arr,ovf_samp,_ = lfilter_with_overflow_monitor(b_q,a_q,x,p)
                stab = sm_ba(a_q)
            else:
                s_q=convert_sos(sos_ref,p)
                h,ms_arr,ovf_samp,_ = sosfilt_with_overflow_monitor(s_q,x,p)
                stab = sm_sos(s_q)

            h = np.array(h, dtype=np.float64)
            mx_h = float(np.max(np.abs(h[np.isfinite(h)]))) if np.any(np.isfinite(h)) else np.inf
            mx_st = float(np.max(ms_arr[np.isfinite(ms_arr)])) if np.any(np.isfinite(ms_arr)) else np.inf

            if not np.any(np.isfinite(h)):
                print(f"  {p:<10}{'OVERFLOW':>12}{'---':>12}"
                      f"{mx_st:>12.2e}{str(ovf_samp):>16}{'FAIL':>8}")
                dtft_results[sid][p] = None
                continue

            H_dtft = np.array([np.sum(h*np.exp(-1j*2*np.pi*f/FS*n_arr)) for f in freqs])
            err = float(np.max(np.abs(np.abs(H_dtft)-H_ref_mag)))
            status = 'OK' if stab > 0 and err < 2.0 else 'DEGRADE'
            ovf_str = str(ovf_samp) if ovf_samp else 'None'
            print(f"  {p:<10}{mx_h:>12.4f}{err:>12.2e}"
                  f"{mx_st:>12.4f}{ovf_str:>16}{status:>8}")
            dtft_results[sid][p] = (freqs, np.abs(H_dtft))

    return dtft_results, freqs, H_ref_mag, w_ref


def test2_sinusoidal():
    print(f"\n{'='*65}")
    print(f"TEST 2: SINUSOIDAL METHOD (AUDIO - Standard Test Tone)")
    print(f"{'='*65}")

    b_ref,a_ref = signal.butter(ORDER_BA, CUTOFF_NORM, output='ba')
    sos_ref     = signal.butter(ORDER_SOS, CUTOFF_NORM, output='sos')
    b_ref=b_ref.astype(np.float64); a_ref=a_ref.astype(np.float64)
    sos_ref=sos_ref.astype(np.float64)

    for label, sid, order in [('BA  (Direct-Form)','ba', ORDER_BA),
                               ('SOS (Ladder)',    'sos',ORDER_SOS)]:
        print(f"\n  {label} order {order}")
        print(f"  {'Freq(Hz)':<10}{'|H|_theory':>11}{'Precision':<10}"
              f"{'|H|_meas':>10}{'err':>10}{'Max_state':>12}{'Overflow':>10}")
        print(f"  {'-'*80}")

        for f_hz in TEST_FREQS_HZ:
            _, H_th = signal.freqz(b_ref, a_ref, worN=[f_hz], fs=FS)
            th = float(np.abs(H_th[0]))
            n = np.arange(N_SINE)
            x = np.sin(2*np.pi*f_hz/FS*n)

            for p in PRECISIONS:
                if sid == 'ba':
                    b_q=convert_ba(b_ref,p); a_q=convert_ba(a_ref,p)
                    y,ms_arr,ovf_s,_ = lfilter_with_overflow_monitor(b_q,a_q,x,p)
                else:
                    s_q=convert_sos(sos_ref,p)
                    y,ms_arr,ovf_s,_ = sosfilt_with_overflow_monitor(s_q,x,p)

                y=np.array(y,dtype=np.float64)
                mx_st = float(np.max(ms_arr[np.isfinite(ms_arr)])) if np.any(np.isfinite(ms_arr)) else np.inf
                ovf = not np.all(np.isfinite(y))
                y_ss=y[N_SETTLE:]; x_ss=x[N_SETTLE:]

                if ovf or not np.any(np.isfinite(y_ss)):
                    print(f"  {f_hz:<10}{th:>11.4f}{p:<10}"
                          f"{'OVERFLOW':>10}{'---':>10}{mx_st:>12.2e}{'YES':>10}")
                    fname=f"sine_{f_hz}Hz_{sid.upper()}_order{order}_{p}.wav"
                    save_wav(os.path.join(RESULTS_DIR,fname), np.zeros(N_SINE))
                    continue

                amp = np.std(y_ss)/(np.std(x_ss)+1e-12)
                err = abs(amp-th)
                print(f"  {f_hz:<10}{th:>11.4f}{p:<10}"
                      f"{amp:>10.4f}{err:>10.2e}{mx_st:>12.4f}{'YES' if ovf else 'NO':>10}")
                fname=f"sine_{f_hz}Hz_{sid.upper()}_order{order}_{p}.wav"
                save_wav(os.path.join(RESULTS_DIR,fname), y)
            print()


def test3_audio():
    print(f"\n{'='*65}")
    print(f"TEST 3: AUDIO SIGNAL TEST (Subjective Listening)")
    print(f"{'='*65}")

    x = speech_signal(N_AUDIO, FS)
    save_wav(os.path.join(RESULTS_DIR,'audio_input.wav'), x)
    print(f"\n  Input: audio_input.wav")
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
                y,ms_arr,ovf_s,_ = lfilter_with_overflow_monitor(b_q,a_q,x,p)
            else:
                s_q=convert_sos(sos_ref,p)
                y,ms_arr,ovf_s,_ = sosfilt_with_overflow_monitor(s_q,x,p)

            y=np.array(y,dtype=np.float64)
            mx_out = float(np.max(np.abs(y[np.isfinite(y)]))) if np.any(np.isfinite(y)) else np.inf
            mx_st  = float(np.max(ms_arr[np.isfinite(ms_arr)])) if np.any(np.isfinite(ms_arr)) else np.inf
            ovf_str = str(ovf_s) if ovf_s else 'None'

            if not np.any(np.isfinite(y)):
                snr_str='OVERFLOW'; y_sv=np.zeros_like(x)
            else:
                noise=y-ref; sp=np.mean(ref**2); np_=np.mean(noise**2)
                snr_db=10*np.log10(sp/np_) if np_>0 else np.inf
                snr_str=f"{snr_db:.1f}" if np.isfinite(snr_db) else "inf"
                y_sv=y

            fname=f"audio_{sid.upper()}_order{order}_{p}.wav"
            save_wav(os.path.join(RESULTS_DIR,fname), y_sv)
            print(f"  {label:<18}{p:<10}{mx_out:>12.4f}"
                  f"{mx_st:>12.4f}{ovf_str:>10}{snr_str:>10}  {fname}")
        print()


def test4_freqz_comparison_plot(dtft_results, freqs_dtft, H_ref_mag, w_ref):
    print(f"\n{'='*65}")
    print(f"TEST 4: FREQUENCY RESPONSE COMPARISON PLOT (AUDIO)")
    print(f"freqz() vs Impulse DTFT vs Sinusoidal method - same axes")
    print(f"{'='*65}")

    b_ref,a_ref = signal.butter(ORDER_BA, CUTOFF_NORM, output='ba')
    sos_ref     = signal.butter(ORDER_SOS, CUTOFF_NORM, output='sos')
    b_ref=b_ref.astype(np.float64); a_ref=a_ref.astype(np.float64)
    sos_ref=sos_ref.astype(np.float64)

    for sid, order, label in [('ba', ORDER_BA,  'Direct-Form (BA)'),
                               ('sos',ORDER_SOS, 'Ladder (SOS)')]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Frequency Response Comparison - {label} order {order}\n'
                     f'freqz() vs Impulse DTFT vs Sinusoidal Method\n'
                     f'fs={FS}Hz  cutoff={CUTOFF_HZ}Hz',
                     fontsize=11, fontweight='bold')

        for idx, p in enumerate(PRECISIONS):
            ax = axes[idx//2][idx%2]
            if sid == 'ba':
                b_q=convert_ba(b_ref,p); a_q=convert_ba(a_ref,p)
                w_q, H_q = signal.freqz(b_q, a_q, worN=1024, fs=FS)
            else:
                s_q=convert_sos(sos_ref,p)
                w_q, H_q = signal.sosfreqz(s_q, worN=1024, fs=FS)

            ax.plot(w_q, 20*np.log10(np.abs(H_q)+1e-300),
                    '-', color='tab:blue', lw=1.5, label='freqz() theoretical', zorder=3)

            if dtft_results.get(sid,{}).get(p) is not None:
                f_dtft, H_dtft_mag = dtft_results[sid][p]
                ax.plot(f_dtft, 20*np.log10(H_dtft_mag+1e-300),
                        '--', color='tab:orange', lw=1.5, label='Impulse DTFT', zorder=2)

            x_pts = []; y_pts = []
            for f_hz in TEST_FREQS_HZ:
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
        fname = os.path.join(RESULTS_DIR, f'freqz_comparison_{sid}_order{order}.png')
        plt.savefig(fname, dpi=120)
        plt.close()
        print(f"  Saved: {fname}")


def deterioration_sweep():
    print(f"\n{'='*65}")
    print(f"DETERIORATION SWEEP (AUDIO)")
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
        b=b.astype(np.float64); a=a.astype(np.float64); sos=sos.astype(np.float64)
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


# PART 2 - VLF RANGE TESTS (fs=1000Hz, cutoff=100Hz)

def vlf_stability_check():
    print(f"\n{'='*65}")
    print(f"VLF STABILITY CHECK")
    print(f"fs={FS_VLF}Hz  cutoff={CUTOFF_VLF}Hz  normalised={CUTOFF_NORM_VLF:.4f}")
    print(f"Applications: power line monitoring, geophysical, submarine comms")
    print(f"{'='*65}")
    print(f"\n{'Order':<7}{'BA_f16':>12}{'BA_f32':>12}{'BA_f64':>12}"
          f"{'SOS_f16':>12}{'SOS_f32':>12}{'SOS_f64':>12}")
    print("-"*75)

    for order in range(2, 21, 2):
        b,a = signal.butter(order, CUTOFF_NORM_VLF, output='ba')
        sos = signal.butter(order, CUTOFF_NORM_VLF, output='sos')
        b=b.astype(np.float64); a=a.astype(np.float64); sos=sos.astype(np.float64)

        row = f"{order:<7}"
        for sid in ['ba','sos']:
            for dt in [np.float16, np.float32, np.float64]:
                if sid == 'ba':
                    sm = sm_ba(a.astype(dt).astype(np.float64))
                else:
                    sm = sm_sos(sos.astype(dt).astype(np.float64))
                flag = '*' if sm < 0 else ' '
                row += f"{sm:>11.4f}{flag}"
        print(row)

    print("* = UNSTABLE")
    print(f"\nBA  order {ORDER_BA_VLF}  - highest stable at float16 (SM=+0.045)")
    print(f"SOS order {ORDER_SOS_VLF} - highest with no float16 coefficient underflow")


def vlf_impulse_test():
    print(f"\n{'='*65}")
    print(f"VLF IMPULSE RESPONSE TEST")
    print(f"fs={FS_VLF}Hz  cutoff={CUTOFF_VLF}Hz")
    print(f"{'='*65}")

    b_ref,a_ref = signal.butter(ORDER_BA_VLF,  CUTOFF_NORM_VLF, output='ba')
    sos_ref     = signal.butter(ORDER_SOS_VLF, CUTOFF_NORM_VLF, output='sos')
    b_ref=b_ref.astype(np.float64); a_ref=a_ref.astype(np.float64)
    sos_ref=sos_ref.astype(np.float64)

    x = np.zeros(N_IMPULSE_VLF); x[0] = 1.0
    w_ref, H_ref = signal.freqz(b_ref, a_ref, worN=N_IMPULSE_VLF//2, fs=FS_VLF)
    H_ref_mag = np.abs(H_ref)
    freqs = np.linspace(0, FS_VLF/2, N_IMPULSE_VLF//2)
    n_arr = np.arange(N_IMPULSE_VLF)

    for label, sid, order in [('BA  (Direct-Form)', 'ba',  ORDER_BA_VLF),
                               ('SOS (Ladder)',      'sos', ORDER_SOS_VLF)]:
        print(f"\n  {label} order {order}")
        print(f"  {'Precision':<10}{'Max|h[n]|':>12}{'DTFT_err':>12}{'SM':>10}{'Status':>8}")
        print(f"  {'-'*55}")

        for p in PRECISIONS:
            if sid == 'ba':
                b_q=convert_ba(b_ref,p); a_q=convert_ba(a_ref,p)
                h = signal.lfilter(b_q, a_q, x)
                stab = sm_ba(a_q)
            else:
                s_q=convert_sos(sos_ref,p)
                h = signal.sosfilt(s_q, x)
                stab = sm_sos(s_q)

            h = np.array(h, dtype=np.float64)
            if not np.any(np.isfinite(h)):
                print(f"  {p:<10}{'OVERFLOW':>12}{'---':>12}{stab:>10.4f}{'FAIL':>8}")
                continue

            mx = float(np.max(np.abs(h[np.isfinite(h)])))
            H_dtft = np.array([np.sum(h*np.exp(-1j*2*np.pi*f/FS_VLF*n_arr))
                                for f in freqs])
            err = float(np.max(np.abs(np.abs(H_dtft)-H_ref_mag)))
            status = 'OK' if stab > 0 and err < 2.0 else 'DEGRADE'
            print(f"  {p:<10}{mx:>12.4f}{err:>12.2e}{stab:>10.4f}{status:>8}")


def vlf_sinusoidal_test():
    print(f"\n{'='*65}")
    print(f"VLF SINUSOIDAL TEST (Standard Test Tone at VLF)")
    print(f"fs={FS_VLF}Hz  cutoff={CUTOFF_VLF}Hz")
    print(f"{'='*65}")

    b_ref,a_ref = signal.butter(ORDER_BA_VLF,  CUTOFF_NORM_VLF, output='ba')
    sos_ref     = signal.butter(ORDER_SOS_VLF, CUTOFF_NORM_VLF, output='sos')
    b_ref=b_ref.astype(np.float64); a_ref=a_ref.astype(np.float64)
    sos_ref=sos_ref.astype(np.float64)

    for label, sid, order in [('BA  (Direct-Form)', 'ba',  ORDER_BA_VLF),
                               ('SOS (Ladder)',      'sos', ORDER_SOS_VLF)]:
        print(f"\n  {label} order {order}")
        print(f"  {'Freq(Hz)':<10}{'|H|_theory':>11}{'Precision':<10}"
              f"{'|H|_meas':>10}{'err':>10}{'Overflow':>10}")
        print(f"  {'-'*65}")

        for f_hz in TEST_FREQS_VLF:
            _, H_th = signal.freqz(b_ref, a_ref, worN=[f_hz], fs=FS_VLF)
            th = float(np.abs(H_th[0]))
            n  = np.arange(N_SINE_VLF)
            x  = np.sin(2*np.pi*f_hz/FS_VLF*n)

            for p in PRECISIONS:
                if sid == 'ba':
                    b_q=convert_ba(b_ref,p); a_q=convert_ba(a_ref,p)
                    y=signal.lfilter(b_q,a_q,x)
                else:
                    s_q=convert_sos(sos_ref,p)
                    y=signal.sosfilt(s_q,x)

                y=np.array(y,dtype=np.float64)
                ovf = not np.all(np.isfinite(y))
                y_ss=y[N_SETTLE_VLF:]; x_ss=x[N_SETTLE_VLF:]

                if ovf or not np.any(np.isfinite(y_ss)):
                    print(f"  {f_hz:<10}{th:>11.4f}{p:<10}"
                          f"{'OVERFLOW':>10}{'---':>10}{'YES':>10}")
                    continue

                amp = np.std(y_ss)/(np.std(x_ss)+1e-12)
                err = abs(amp-th)
                print(f"  {f_hz:<10}{th:>11.4f}{p:<10}"
                      f"{amp:>10.4f}{err:>10.2e}{'NO':>10}")
            print()


def vlf_signal_test():
    print(f"\n{'='*65}")
    print(f"VLF SIGNAL TEST (Power Line / Geophysical Signal)")
    print(f"fs={FS_VLF}Hz  cutoff={CUTOFF_VLF}Hz  duration={N_AUDIO_VLF/FS_VLF:.1f}s")
    print(f"{'='*65}")

    x = vlf_signal(N_AUDIO_VLF, FS_VLF)
    save_wav(os.path.join(RESULTS_DIR,'vlf_input.wav'), x, fs=FS_VLF)
    print(f"\n  Input: vlf_input.wav")
    print(f"  Content: 50Hz power line + harmonics + 10/30Hz + 1Hz geophysical")

    b_ref,a_ref = signal.butter(ORDER_BA_VLF,  CUTOFF_NORM_VLF, output='ba')
    sos_ref     = signal.butter(ORDER_SOS_VLF, CUTOFF_NORM_VLF, output='sos')
    b_ref=b_ref.astype(np.float64); a_ref=a_ref.astype(np.float64)
    sos_ref=sos_ref.astype(np.float64)

    y_ref_ba  = signal.lfilter(b_ref, a_ref, x)
    y_ref_sos = signal.sosfilt(sos_ref, x)
    save_wav(os.path.join(RESULTS_DIR,'vlf_BA_float64_ref.wav'),  y_ref_ba,  fs=FS_VLF)
    save_wav(os.path.join(RESULTS_DIR,'vlf_SOS_float64_ref.wav'), y_ref_sos, fs=FS_VLF)

    print(f"\n  {'Structure':<18}{'Precision':<10}{'Max output':>12}"
          f"{'SNR(dB)':>10}{'Overflow':>10}  File")
    print(f"  {'-'*75}")

    for label, sid, order, ref in [
            ('BA  (Direct-Form)', 'ba',  ORDER_BA_VLF,  y_ref_ba),
            ('SOS (Ladder)',      'sos', ORDER_SOS_VLF, y_ref_sos)]:
        for p in PRECISIONS:
            if sid == 'ba':
                b_q=convert_ba(b_ref,p); a_q=convert_ba(a_ref,p)
                y=signal.lfilter(b_q,a_q,x)
            else:
                s_q=convert_sos(sos_ref,p)
                y=signal.sosfilt(s_q,x)

            y=np.array(y,dtype=np.float64)
            ovf = not np.all(np.isfinite(y))
            mx = float(np.max(np.abs(y[np.isfinite(y)]))) if np.any(np.isfinite(y)) else np.inf

            if ovf:
                snr_str='OVERFLOW'; y_sv=np.zeros_like(x)
            else:
                noise=y-ref; sp=np.mean(ref**2); np_=np.mean(noise**2)
                snr_db=10*np.log10(sp/np_) if np_>0 else np.inf
                snr_str=f"{snr_db:.1f}" if np.isfinite(snr_db) else "inf"
                y_sv=y

            fname=f"vlf_{sid.upper()}_order{order}_{p}.wav"
            save_wav(os.path.join(RESULTS_DIR,fname), y_sv, fs=FS_VLF)
            print(f"  {label:<18}{p:<10}{mx:>12.4f}"
                  f"{snr_str:>10}{'YES' if ovf else 'NO':>10}  {fname}")
        print()


def vlf_response_plot():
    print(f"\n{'='*65}")
    print(f"VLF FREQUENCY RESPONSE PLOT")
    print(f"{'='*65}")

    b_ref,a_ref = signal.butter(ORDER_BA_VLF,  CUTOFF_NORM_VLF, output='ba')
    sos_ref     = signal.butter(ORDER_SOS_VLF, CUTOFF_NORM_VLF, output='sos')
    b_ref=b_ref.astype(np.float64); a_ref=a_ref.astype(np.float64)
    sos_ref=sos_ref.astype(np.float64)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for p in PRECISIONS:
        style = '--' if p == 'float16' else '-'
        lw = 2 if p in ('float16','float32') else 1.2

        b_q=convert_ba(b_ref,p); a_q=convert_ba(a_ref,p)
        w,H=signal.freqz(b_q,a_q,worN=1024,fs=FS_VLF)
        ax1.plot(w, 20*np.log10(np.abs(H)+1e-300),
                 style, color=COLORS[p], lw=lw, label=p)

        sos_q=convert_sos(sos_ref,p)
        w,H=signal.sosfreqz(sos_q,worN=1024,fs=FS_VLF)
        ax2.plot(w, 20*np.log10(np.abs(H)+1e-300),
                 style, color=COLORS[p], lw=lw, label=p)

    for ax, title, order in [
            (ax1, f'Direct-Form (BA) order {ORDER_BA_VLF}',  ORDER_BA_VLF),
            (ax2, f'Ladder (SOS) order {ORDER_SOS_VLF}',     ORDER_SOS_VLF)]:
        ax.set_ylim(-100, 20)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('|H(f)| (dB)')
        ax.set_title(f'VLF - {title}\nfs={FS_VLF}Hz  cutoff={CUTOFF_VLF}Hz')
        ax.axvline(CUTOFF_VLF, color='gray', ls=':', lw=1)
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('VLF Filter Comparison - Direct-Form vs Ladder\n'
                 f'fs={FS_VLF}Hz  cutoff={CUTOFF_VLF}Hz',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, 'vlf_response_comparison.png')
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Saved: {fname}")


# Main
if __name__ == '__main__':
    print("COMPLETE AUDIO AND VLF FILTER TEST")
    print("Student: Darshan Mahadeva Naika | A00090581 | EEN1095")
    print("Project: Design and Evaluation of Low-Sensitivity Digital Filters")
    print("         Implemented in Python Under Varying Numerical Precision Levels")

    # PART 1: AUDIO RANGE
    print(f"\n{'#'*65}")
    print(f"# PART 1: AUDIO RANGE (fs={FS}Hz, cutoff={CUTOFF_HZ}Hz)")
    print(f"# BA order={ORDER_BA} | SOS order={ORDER_SOS}")
    print(f"{'#'*65}")

    check_dynamic_range()
    dtft_res, freqs_d, H_ref_mag, w_ref = test1_impulse()
    test2_sinusoidal()
    test3_audio()
    test4_freqz_comparison_plot(dtft_res, freqs_d, H_ref_mag, w_ref)
    deterioration_sweep()

    # PART 2: VLF RANGE
    print(f"\n{'#'*65}")
    print(f"# PART 2: VLF RANGE (fs={FS_VLF}Hz, cutoff={CUTOFF_VLF}Hz)")
    print(f"# Applications: power line, geophysical, submarine comms")
    print(f"# BA order={ORDER_BA_VLF} | SOS order={ORDER_SOS_VLF}")
    print(f"{'#'*65}")

    vlf_stability_check()
    vlf_impulse_test()
    vlf_sinusoidal_test()
    vlf_signal_test()
    vlf_response_plot()

    # Summary - dynamically compute key SNR values and deterioration points
    # Audio range SNR
    x_audio = speech_signal(N_AUDIO, FS)
    b_ref_a, a_ref_a = signal.butter(ORDER_BA,  CUTOFF_NORM, output='ba')
    sos_ref_a        = signal.butter(ORDER_SOS, CUTOFF_NORM, output='sos')
    b_ref_a=b_ref_a.astype(np.float64); a_ref_a=a_ref_a.astype(np.float64)
    sos_ref_a=sos_ref_a.astype(np.float64)
    y_ref_ba_a  = signal.lfilter(b_ref_a, a_ref_a, x_audio)
    y_ref_sos_a = signal.sosfilt(sos_ref_a, x_audio)

    def calc_snr(y, ref):
        y64 = np.array(y, dtype=np.float64)
        if not np.all(np.isfinite(y64)): return None
        noise = y64 - ref
        sp = np.mean(ref**2); np_ = np.mean(noise**2)
        return 10*np.log10(sp/np_) if np_ > 0 else np.inf

    ba_f16_a  = signal.lfilter(convert_ba(b_ref_a,'float16'), convert_ba(a_ref_a,'float16'), x_audio)
    sos_f16_a = signal.sosfilt(convert_sos(sos_ref_a,'float16'), x_audio)
    snr_ba_f16_audio  = calc_snr(ba_f16_a,  y_ref_ba_a)
    snr_sos_f16_audio = calc_snr(sos_f16_a, y_ref_sos_a)

    # VLF range SNR
    x_vlf = vlf_signal(N_AUDIO_VLF, FS_VLF)
    b_ref_v, a_ref_v = signal.butter(ORDER_BA_VLF,  CUTOFF_NORM_VLF, output='ba')
    sos_ref_v        = signal.butter(ORDER_SOS_VLF, CUTOFF_NORM_VLF, output='sos')
    b_ref_v=b_ref_v.astype(np.float64); a_ref_v=a_ref_v.astype(np.float64)
    sos_ref_v=sos_ref_v.astype(np.float64)
    y_ref_ba_v  = signal.lfilter(b_ref_v, a_ref_v, x_vlf)
    y_ref_sos_v = signal.sosfilt(sos_ref_v, x_vlf)

    ba_f16_v  = signal.lfilter(convert_ba(b_ref_v,'float16'), convert_ba(a_ref_v,'float16'), x_vlf)
    sos_f16_v = signal.sosfilt(convert_sos(sos_ref_v,'float16'), x_vlf)
    snr_ba_f16_vlf  = calc_snr(ba_f16_v,  y_ref_ba_v)
    snr_sos_f16_vlf = calc_snr(sos_f16_v, y_ref_sos_v)

    # Deterioration points - find order where SNR drops below 40 dB
    x_det = speech_signal(FS*2, FS)
    SNR_THRESHOLD = 40.0
    ba_degrade_order  = None
    sos_degrade_order = None
    for order in range(2, 23, 2):
        b_d,a_d = signal.butter(order, CUTOFF_NORM, output='ba')
        sos_d   = signal.butter(order, CUTOFF_NORM, output='sos')
        b_d=b_d.astype(np.float64); a_d=a_d.astype(np.float64); sos_d=sos_d.astype(np.float64)
        ref_ba_d  = signal.lfilter(b_d, a_d, x_det)
        ref_sos_d = signal.sosfilt(sos_d, x_det)
        if ba_degrade_order is None:
            y_ba_d = signal.lfilter(b_d.astype(np.float16).astype(np.float64),
                                     a_d.astype(np.float16).astype(np.float64), x_det)
            snr_d = calc_snr(y_ba_d, ref_ba_d)
            if snr_d is None or snr_d < SNR_THRESHOLD:
                ba_degrade_order = order
        if sos_degrade_order is None:
            y_sos_d = signal.sosfilt(sos_d.astype(np.float16).astype(np.float64), x_det)
            snr_d2 = calc_snr(y_sos_d, ref_sos_d)
            if snr_d2 is None or snr_d2 < SNR_THRESHOLD:
                sos_degrade_order = order

    # Format SNR strings dynamically
    def fmt_snr(snr):
        if snr is None: return "OVERFLOW"
        return f"{snr:.1f} dB"

    audio_improvement = (snr_sos_f16_audio - snr_ba_f16_audio) \
                        if snr_ba_f16_audio is not None and snr_sos_f16_audio is not None else None
    vlf_improvement   = (snr_sos_f16_vlf - snr_ba_f16_vlf) \
                        if snr_ba_f16_vlf is not None and snr_sos_f16_vlf is not None else None

    print(f"\n{'='*65}")
    print(f"ALL TESTS COMPLETE")
    print(f"Results saved to: {os.path.abspath(RESULTS_DIR)}/")
    print(f"\nAUDIO RANGE KEY RESULTS (dynamically computed):")
    print(f"  BA  float16 SNR = {fmt_snr(snr_ba_f16_audio)}")
    print(f"  SOS float16 SNR = {fmt_snr(snr_sos_f16_audio)}"
          + (f"  (+{audio_improvement:.1f} dB improvement)" if audio_improvement else ""))
    print(f"  BA  deteriorates at order {ba_degrade_order}  (float16, SNR < {SNR_THRESHOLD} dB)")
    print(f"  SOS deteriorates at order {sos_degrade_order} (float16, SNR < {SNR_THRESHOLD} dB)")
    print(f"\nVLF RANGE KEY RESULTS (dynamically computed):")
    print(f"  BA  float16 SNR = {fmt_snr(snr_ba_f16_vlf)}"
          + ("  (OVERFLOW - unusable)" if snr_ba_f16_vlf is None else ""))
    print(f"  SOS float16 SNR = {fmt_snr(snr_sos_f16_vlf)}"
          + (f"  (+{vlf_improvement:.1f} dB improvement)" if vlf_improvement else ""))
    print(f"\nLISTENING GUIDE (AUDIO):")
    print(f"  audio_input.wav                <- original signal")
    print(f"  audio_BA_float64_ref.wav       <- BA reference (clean)")
    print(f"  audio_BA_order{ORDER_BA}_float16.wav <- BA float16 (distorted?)")
    print(f"  audio_SOS_float64_ref.wav      <- SOS reference (clean)")
    print(f"  audio_SOS_order{ORDER_SOS}_float16.wav <- SOS float16 (better?)")
    print(f"\nLISTENING GUIDE (VLF):")
    print(f"  vlf_input.wav                  <- VLF signal (power line sim)")
    print(f"  vlf_BA_float64_ref.wav         <- BA VLF reference")
    print(f"  vlf_BA_order{ORDER_BA_VLF}_float16.wav  <- BA VLF float16 (distorted)")
    print(f"  vlf_SOS_float64_ref.wav        <- SOS VLF reference")
    print(f"  vlf_SOS_order{ORDER_SOS_VLF}_float16.wav <- SOS VLF float16 (better)")
