"""Classical thin-airfoil theory (Glauert) as a physics baseline."""
import numpy as np

from .geometry import camber_line


def thin_airfoil(m, p, alpha_deg, n=2000):
    """Return (cl, cm_c/4) from thin-airfoil theory. Ignores thickness entirely."""
    theta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(theta))
    _, dz = camber_line(x, m, p)
    a = np.radians(alpha_deg)
    A0 = a - np.trapezoid(dz, theta) / np.pi
    A1 = 2 / np.pi * np.trapezoid(dz * np.cos(theta), theta)
    A2 = 2 / np.pi * np.trapezoid(dz * np.cos(2 * theta), theta)
    cl = np.pi * (2 * A0 + A1)
    cm = np.pi / 4 * (A2 - A1)
    return cl, cm
