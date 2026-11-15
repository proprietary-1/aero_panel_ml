"""Hess-Smith panel method: constant-strength source panels + uniform vortex.

Inviscid, incompressible 2D potential flow with a Kutta condition. Gives
pressure distribution, lift and pitching moment for any closed airfoil.
"""
from dataclasses import dataclass

import numpy as np

TWO_PI = 2 * np.pi


@dataclass
class PanelResult:
    cl: float          # lift coefficient (from pressure integration)
    cm: float          # pitching moment about c/4, nose-up positive
    cp_min: float      # peak suction (most negative Cp)
    cl_circ: float     # lift from Kutta-Joukowski (cross-check)
    xc: np.ndarray     # control-point x
    yc: np.ndarray     # control-point y
    cp: np.ndarray     # Cp at control points


def _influence(x, y):
    xa, ya, xb, yb = x[:-1], y[:-1], x[1:], y[1:]
    dx, dy = xb - xa, yb - ya
    length = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    xc, yc = 0.5 * (xa + xb), 0.5 * (ya + yb)

    ax, ay = xa[None, :] - xc[:, None], ya[None, :] - yc[:, None]
    bx, by = xb[None, :] - xc[:, None], yb[None, :] - yc[:, None]
    r1, r2 = np.hypot(ax, ay), np.hypot(bx, by)
    beta = np.arctan2(ax * by - ay * bx, ax * bx + ay * by)  # angle subtended
    n = len(th)
    beta[np.diag_indices(n)] = np.pi
    with np.errstate(divide="ignore"):
        lnr = np.log(r2 / r1)
    lnr[np.diag_indices(n)] = 0.0

    dth = th[:, None] - th[None, :]
    s, c = np.sin(dth), np.cos(dth)
    an = (s * lnr + c * beta) / TWO_PI   # source -> normal velocity
    at = (s * beta - c * lnr) / TWO_PI   # source -> tangential velocity
    return an, at, th, length, xc, yc


def solve(x, y, alpha_deg):
    """Solve the flow around airfoil nodes (x, y) at angle of attack alpha (deg)."""
    a = np.radians(alpha_deg)
    an, at, th, length, xc, yc = _influence(x, y)
    n = len(th)

    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = an
    A[:n, n] = -at.sum(axis=1)                 # vortex -> normal velocity
    A[n, :n] = at[0] + at[-1]                   # Kutta: Vt_first + Vt_last = 0
    A[n, n] = an[0].sum() + an[-1].sum()
    rhs = np.empty(n + 1)
    rhs[:n] = -np.sin(a - th)
    rhs[n] = -(np.cos(a - th[0]) + np.cos(a - th[-1]))
    sol = np.linalg.solve(A, rhs)
    q, gamma = sol[:n], sol[n]

    vt = at @ q + gamma * an.sum(axis=1) + np.cos(a - th)
    cp = 1 - vt**2

    # outward normals for clockwise ordering
    nx, ny = -np.sin(th), np.cos(th)
    fx = -np.sum(cp * length * nx)
    fy = -np.sum(cp * length * ny)
    cl = fy * np.cos(a) - fx * np.sin(a)
    cm = -np.sum(-cp * length * ((xc - 0.25) * ny - yc * nx))
    cl_circ = 2 * gamma * length.sum()
    return PanelResult(cl, cm, cp.min(), cl_circ, xc, yc, cp)
