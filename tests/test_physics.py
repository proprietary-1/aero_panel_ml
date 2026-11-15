"""Physics and gradient checks. Run with:  python tests/test_physics.py  (or pytest)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aeroml.geometry import naca4
from aeroml.mlp import MLP
from aeroml.panel import solve
from aeroml.thin_airfoil import thin_airfoil


def test_symmetric_airfoil_zero_lift():
    r = solve(*naca4(0, 0.4, 0.12), 0.0)
    assert abs(r.cl) < 1e-8 and abs(r.cm) < 1e-8


def test_lift_slope_matches_theory():
    g = naca4(0, 0.4, 0.12, 200)
    slope = (solve(*g, 4).cl - solve(*g, -4).cl) / np.radians(8)
    assert abs(slope / (2 * np.pi * (1 + 0.77 * 0.12)) - 1) < 0.02


def test_pressure_and_circulation_lift_agree():
    r = solve(*naca4(0.04, 0.4, 0.12, 200), 6)
    assert abs(r.cl - r.cl_circ) / r.cl < 0.01


def test_camber_moment_matches_thin_airfoil():
    r = solve(*naca4(0.02, 0.4, 0.06, 200), 2)
    assert abs(r.cm - thin_airfoil(0.02, 0.4, 2)[1]) < 0.006


def test_mlp_input_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    X = rng.random((200, 4)); Y = np.column_stack([np.sin(X @ [1, 2, 3, 4]), X[:, 0] ** 2])
    net = MLP([4, 16, 2]); net.fit(X, Y, X, Y, epochs=5, verbose=False)
    x, dy = X[:1], np.array([[0.7, -1.3]])
    g = net.vjp_input(x, dy)[0]
    eps = 1e-6
    fd = [((net.predict(x + eps * e) - net.predict(x - eps * e)) @ dy[0])[0] / (2 * eps) for e in np.eye(4)]
    assert np.allclose(g, fd, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
