# Elliptic Filter Design — Library and Source Code Notes

Student: Darshan Mahadeva Naika | A00090581 | EEN1095


## 1. Library Used

All elliptic (Cauer) filters in this project are designed using:

```python
from scipy import signal
b, a = signal.ellip(N, rp=1.0, rs=40.0, Wn=0.3, btype='low', output='ba')
```

- **Package:** SciPy `scipy.signal` (SciPy 1.17.1 used here)
- **Reference:** Virtanen, P. et al. (2020). "SciPy 1.0: Fundamental algorithms for
  scientific computing in Python." *Nature Methods*, 17, 261-272. [Ref 5 in project bibliography]
- **Source file (in SciPy install):**
  `scipy/signal/_filter_design.py`

`ellip()` is a thin wrapper that calls `iirfilter()`, which in turn calls the
analog elliptic prototype function `ellipap()`. **`ellipap()` is where the
actual elliptic-filter mathematics lives.**

---

## 2. What "Elliptic" Means Mathematically

An elliptic (Cauer) filter is the optimal IIR filter design for a given
order: for a specified passband ripple (`rp`, dB) and stopband attenuation
(`rs`, dB), it achieves the **narrowest possible transition band**. This is
why elliptic filters are the most aggressive/sensitive filter type in this
project — they pack the poles and zeros as tightly as possible near the
unit circle to get the sharpest roll-off.

The pole/zero locations are not given by simple closed-form trigonometric
expressions (as for Butterworth/Chebyshev). Instead they require:

- **Complete elliptic integrals of the first kind**, `K(m)` — computed via
  `scipy.special.ellipk` and `scipy.special.ellipkm1`
- **Jacobian elliptic functions** `sn`, `cn`, `dn` — computed via
  `scipy.special.ellipj`
- An iterative **"degree equation"** solve (a nome/theta-function series)
  to find the elliptic modulus `m` that satisfies the ripple/attenuation
  specification for the given order `N`

These are the same special functions used in the theory of elliptic
integrals/elliptic curves — hence "elliptic filter."


## 3. The Core Source Code — `ellipap()`

This is the actual SciPy function (from `scipy/signal/_filter_design.py`)
that computes the analog elliptic low-pass prototype poles `p`, zeros `z`,
and gain `k`:

```python
def ellipap(N, rp, rs):
    """Return (z,p,k) of Nth-order elliptic analog lowpass filter."""

    eps_sq = _pow10m1(0.1 * rp)          # ripple parameter  eps^2 = 10^(rp/10) - 1
    eps = sqrt(eps_sq)

    ck1_sq = eps_sq / _pow10m1(0.1 * rs) # selectivity parameter
    if ck1_sq == 0:
        raise ValueError("Cannot design a filter with given rp and rs"
                          " specifications.")

    m = _ellipdeg(N, ck1_sq)             # solve the "degree equation" for modulus m
    capk = special.ellipk(m)             # complete elliptic integral K(m)

    j = np.arange(1 - N % 2, N, 2)
    [s, c, d, phi] = special.ellipj(j * capk / N, m * np.ones(len(j)))

    # --- zeros: come from sn() ---
    snew = np.compress(abs(s) > EPSILON, s, axis=-1)
    z = 1j / (sqrt(m) * snew)
    z = np.concatenate((z, np.conjugate(z)))

    # --- poles: from sn/cn/dn at a shifted argument v0 ---
    r = _arc_jac_sc1(1. / eps, ck1_sq)
    v0 = capk * r / (N * special.ellipk(ck1_sq))
    [sv, cv, dv, phi] = special.ellipj(v0, 1 - m)
    p = -(c * d * sv * cv + 1j * s * dv) / (1 - (d * sv) ** 2.0)

    if N % 2:
        newp = np.compress(abs(p.imag) > EPSILON * sqrt(sum(p*conj(p)).real),
                            p, axis=-1)
        p = np.concatenate((p, np.conjugate(newp)))
    else:
        p = np.concatenate((p, np.conjugate(p)))

    k = (np.prod(-p) / np.prod(-z)).real
    if N % 2 == 0:
        k = k / sqrt(1 + eps_sq)

    return z, p, k
```

*(condensed — full version with docstring and edge cases for N=0,1 is in
`scipy/signal/_filter_design.py`)*

### Key helper: `_ellipdeg()` — solving the degree equation

```python
def _ellipdeg(n, m1):
    """Solve n * K(m)/K'(m) = K1(m1)/K1'(m1) for m, via nome series."""
    K1  = special.ellipk(m1)
    K1p = special.ellipkm1(m1)

    q1 = np.exp(-np.pi * K1p / K1)
    q  = q1 ** (1 / n)

    mnum = np.arange(_ELLIPDEG_MMAX + 1)
    mden = np.arange(1, _ELLIPDEG_MMAX + 2)

    num = np.sum(q ** (mnum * (mnum + 1)))
    den = 1 + 2 * np.sum(q ** (mden ** 2))

    return 16 * q * (num / den) ** 4
```

This is the part of the mathematics that's genuinely non-trivial: it solves
a transcendental equation relating the filter order `N` to the elliptic
modulus `m` using a **nome (q-series) expansion** — an infinite series
truncated at `_ELLIPDEG_MMAX` terms.

---

## 4. After `ellipap()` — Digital Conversion

Once `ellipap()` returns the analog prototype `(z, p, k)`, `iirfilter()`:

1. Frequency-scales the prototype to the requested cutoff `Wn=0.3`
2. Applies the **bilinear transform** to map the analog s-plane prototype
   to the digital z-plane
3. Converts pole/zero/gain form to numerator/denominator (`b`, `a`)
   polynomials via `zpk2tf()`

This `(b, a)` pair is what is passed into this project's `convert()` and
`stability_margin()` / `freqz()` functions for the precision-sensitivity
analysis.

---

## 5. SciPy's Own References for This Algorithm

SciPy's docstring for `ellipap()` cites:

1. Lutovac, Tosic, and Evans, *Filter Design for Signal Processing*,
   Chapters 5 and 12.
2. Orfanidis, *Lecture Notes on Elliptic Filter Design*,
   https://www.ece.rutgers.edu/~orfanidi/ece521/notes.pdf

These are the same texts the elliptic integral / nome-series formulation
(`_ellipdeg`, `_arc_jac_sc1`) is derived from.

---

## 6. Where to See the Full Source

To view the complete, unmodified source on your own machine:

```python
import inspect
from scipy.signal import ellip
from scipy.signal._filter_design import ellipap, _ellipdeg, _arc_jac_sc1
print(inspect.getsource(ellip))      # wrapper / docstring
print(inspect.getsource(ellipap))    # core elliptic prototype math
print(inspect.getsource(_ellipdeg))  # degree-equation solver
```

Or browse directly on GitHub:
https://github.com/scipy/scipy/blob/main/scipy/signal/_filter_design.py

---

## Summary

- **Library:** `scipy.signal.ellip()` (SciPy, Virtanen et al. 2020 [Ref 5])
- **Core math location:** `scipy/signal/_filter_design.py`, function `ellipap()`
- **Underlying special functions:** `scipy.special.ellipj/ellipk/ellipkm1`
  (Jacobian elliptic functions and complete elliptic integrals)
- **Mathematical references SciPy itself cites:** Lutovac/Tosic/Evans
  (textbook) and Orfanidis (lecture notes) — both standard elliptic filter
  design texts
- This is why Elliptic is the most sensitive filter type in Steps 2 and B —
  the poles/zeros from `ellipap()` are placed as close to the unit circle
  as the order allows, by construction.
