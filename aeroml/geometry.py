"""NACA 4-digit airfoil geometry."""
import numpy as np


def camber_line(x, m, p):
    """Mean camber line y_c(x) and slope dy_c/dx for a NACA 4-digit section."""
    x = np.asarray(x, dtype=float)
    if m <= 0 or p <= 0:
        return np.zeros_like(x), np.zeros_like(x)
    fwd = x < p
    yc = np.where(fwd, m / p**2 * (2 * p * x - x**2),
                  m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2))
    dyc = np.where(fwd, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    return yc, dyc


def naca4(m, p, t, n_panels=140):
    """Panel nodes for a NACA 4-digit airfoil (closed trailing edge, cosine spacing).

    m: max camber (fraction of chord), p: camber position, t: thickness.
    Nodes run clockwise: trailing edge -> lower surface -> leading edge ->
    upper surface -> trailing edge. Returns (x, y) with n_panels + 1 points.
    """
    if n_panels % 2:
        raise ValueError("n_panels must be even")
    beta = np.linspace(0.0, np.pi, n_panels // 2 + 1)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1036 * x**4)
    yc, dyc = camber_line(x, m, p)
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    xs = np.concatenate([xl[::-1], xu[1:]])
    ys = np.concatenate([yl[::-1], yu[1:]])
    return xs, ys
