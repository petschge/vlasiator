"""
Dispersion relation of the electrostatic electron Bernstein modes propagating
exactly perpendicular to B0, electron terms only, ions stationary based onxi
Eq.  (51) of P. Kilian, P. A. Munoz, C. Schreiner, F. Spanier,
"Plasma Waves as a Benchmark Problem", arXiv:1611.01127

    1 - (2 * Wce^2) / (k^2 * lambda_D^2) * exp(-lambda_e)
          * sum_{n=1}^{infty} n^2 * I_n(lambda_e) / (omega^2 - n^2 * Wce^2) = 0

with

    lambda_e = k^2 * r_e^2       (Eq. 49)
    r_e      = v_th,e / Wce      (Eq. 48, electron gyro radius)
    lambda_D = v_th,e / w_pe     (Eq. 46, electron Debye length)

I_n is the modified Bessel function of the first kind of integer order n
"""

import numpy as np
from scipy.special import ive
from scipy.optimize import brentq


def bernstein_lhs(omega, k, Wce, lambda_D, r_e, order):
    """
    Evaluate the left-hand side of Eq. (51) at a single frequency, with
    the infinite sum over gyro-harmonics n truncated to n = 1 .. order.

    Parameters
    ----------
    omega : Angular frequency at which to evaluate the LHS.
    k : Perpendicular wavenumber (must be nonzero; units consistent with
        r_e and lambda_D, e.g. both in cm if k is in 1/cm).
    Wce : Electron gyro frequency
    lambda_D : Debye length
    r_e : Electron gyro radius
    order : Number of terms n = 1 .. order kept from the infinite sum.

    Returns
    -------
    LHS of Eq. (51) at omega. Returns NaN if omega falls on (or is
    numerically indistinguishable from) a resonance omega = n * Wce,
    where the expression is singular.
    """
    n = np.arange(1, order + 1, dtype=float)

    lambda_e = k**2 * r_e**2

    # exp(-lambda_e) * I_n(lambda_e), computed directly via the exponentially
    # scaled Bessel function `ive` in a single numerically stable step.
    bessel_term = ive(n, lambda_e)  # shape (order,)
    denom = omega**2 - (n * Wce) ** 2  # shape (order,)

    # Flag terms whose pole sits essentially exactly at this omega, so we
    # don't divide by (near) zero.
    scale = (n * Wce) ** 2
    tol = np.finfo(float).eps * np.maximum(scale, 1.0) * 100
    singular = np.abs(denom) < tol

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = n**2 * bessel_term / denom
    terms = np.where(singular, np.nan, terms)

    total = np.sum(terms)
    prefactor = (2.0 * Wce**2) / (k**2 * lambda_D**2)
    return float(1.0 - prefactor * total)


def bernstein_root_in_bracket(k, i, Wce, lambda_D, r_e, order=None):
    """
    This dispersion relation's poles sit at exactly omega = n * Wce. So lets
    find the single root inside the specific band between (i * Wce, (i+1) *
    Wce) for a fixed k. Note that the LHS always approaches -infinity just
    above the lower pole i*Wce and +infinity just below the upper pole
    (i+1)*Wce, for any k > 0 -- so a root is guaranteed to exist strictly
    inside the band

    This function starts at the initial guess omega0 = (i + 0.5) * Wce (the
    midpoint of the band) and, using the sign of f(omega0) to pick which pole
    must bound the root, repeatedly halves the remaining distance to that pole
    until a genuine sign change relative to f(omega0) is found -- then hands
    that bracket to `brentq`. If no sign change can be resolved before the step
    underflows (double-precision floor reached), the branch's frequency is, at
    this k, numerically indistinguishable from the bounding harmonic itself,
    and that pole.  is returned.

    Parameters
    ----------
    k : Perpendicular wavenumber (must be nonzero).
    i : Bracket index; searches the band (i*Wce, (i+1)*Wce).
    Wce : Electron gyro frequency
    lambda_D : Debye length
    r_e : Electron gyro radius
    order : optional number of terms n = 1 .. order kept in the truncated sum. Defaults to 2 * i

    Returns
    -------
    The root omega in (i*Wce, (i+1)*Wce). If no sign change can be
    resolved precision, the bounding pole itself is returned.
    """

    if order is None:
        order = 2 * i
    if order < i + 1:
        raise ValueError(f"order must be >= i + 1 = {i + 1} to include this band's upper pole")

    lo_pole = i * Wce
    hi_pole = (i + 1) * Wce

    def f(w):
        return bernstein_lhs(w, k, Wce, lambda_D, r_e, order)

    x0 = (i + 0.5) * Wce
    f0 = f(x0)
    if f0 == 0.0:
        return x0

    target_pole = lo_pole if f0 > 0.0 else hi_pole

    gap = target_pole - x0
    bracket_point = None
    for _ in range(64): # no more than 64 bisections
        gap *= 0.5
        x = target_pole - gap
        if x == target_pole:
            # x is no longer distinguishable from the pole itself in double precision.
            break
        fx = f(x)
        if not np.isfinite(fx):
            break
        if fx * f0 < 0.0:
            bracket_point = x
            break

    if bracket_point is None:
        return target_pole

    lo, hi = sorted((x0, bracket_point))
    return brentq(f, lo, hi)


def find_bernstein_mode_branch(i, kmax, Wce, lambda_D, r_e, ksamples=200, order=10):
    """
    Trace the i-th electron Bernstein mode branch
    in the band (i*Wce, (i+1)*Wce) across k in (0, kmax], sampled at
    `ksamples` points, by calling `bernstein_root_in_bracket` once per k.

    Parameters
    ----------
    i : Bracket index; traces the band (i*Wce, (i+1)*Wce).
    kmax : Largest wavenumber scanned
    Wce, lambda_D, r_e, order : See `bernstein_root_in_bracket`.
    ksamples : Number of k values sampled in (0, kmax].

    Returns
    -------
    the k grid and the corresponding root of Eq. (51) in band i at each k. At
    any k where the root could not be resolved in double precision, the entry
    in `omega_values` is instead the bounding pole itself
    """

    if kmax <= 0:
        raise ValueError("kmax must be > 0")
    if ksamples < 1:
        raise ValueError("ksamples must be >= 1")

    k_values = np.linspace(0.0, kmax, ksamples + 1)[1:]
    omega_values = np.array([
        bernstein_root_in_bracket(k, i, Wce, lambda_D, r_e, order=order)
        for k in k_values
    ])
    return k_values, omega_values
