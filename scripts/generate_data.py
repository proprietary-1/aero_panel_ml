"""Step 2: build a training set by running the panel solver on sampled designs.

Design space (Latin hypercube):
    m     max camber        [0.00, 0.06]
    p     camber position   [0.20, 0.60]
    t     thickness         [0.06, 0.18]
    alpha angle of attack   [-4, 12] deg
Outputs: Cl, Cm_c/4, Cp_min (peak suction).
"""
import time

from _path import ROOT
import numpy as np

from aeroml.geometry import naca4
from aeroml.panel import solve

BOUNDS = np.array([[0.00, 0.06], [0.20, 0.60], [0.06, 0.18], [-4.0, 12.0]])
N_SAMPLES = 4000
N_PANELS = 140


def latin_hypercube(n, bounds, rng):
    d = len(bounds)
    u = (np.argsort(rng.random((n, d)), axis=0) + rng.random((n, d))) / n
    return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])


def main():
    rng = np.random.default_rng(42)
    X = latin_hypercube(N_SAMPLES, BOUNDS, rng)
    Y = np.empty((N_SAMPLES, 3))
    t0 = time.time()
    for i, (m, p, t, a) in enumerate(X):
        r = solve(*naca4(m, p, t, N_PANELS), a)
        Y[i] = r.cl, r.cm, r.cp_min
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{N_SAMPLES} solved ({time.time() - t0:.1f}s)")
    out = ROOT / "data" / "dataset.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez(out, X=X, Y=Y, inputs=["m", "p", "t", "alpha"], outputs=["cl", "cm", "cp_min"])
    print(f"saved {out.relative_to(ROOT)}: {N_SAMPLES} samples in {time.time() - t0:.1f}s")
    for name, col in zip(["Cl", "Cm", "Cp_min"], Y.T):
        print(f"  {name:6s} range [{col.min():+.3f}, {col.max():+.3f}]")


if __name__ == "__main__":
    main()
