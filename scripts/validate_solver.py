"""Step 1: check the panel solver against theory and for grid convergence."""
from _path import ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aeroml.geometry import naca4
from aeroml.panel import solve
from aeroml.thin_airfoil import thin_airfoil

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

# 1. Lift curve: panel method vs thin-airfoil theory
alphas = np.linspace(-4, 12, 17)
for (m, p, t), col in [((0, 0.4, 0.12), "C0"), ((0.02, 0.4, 0.12), "C1"), ((0.04, 0.4, 0.12), "C2")]:
    g = naca4(m, p, t)
    cl = [solve(*g, a).cl for a in alphas]
    tat = [thin_airfoil(m, p, a)[0] for a in alphas]
    name = f"NACA {round(m*100)}{round(p*10) if m > 0 else 0}{round(t*100):02d}"
    ax[0].plot(alphas, cl, "o-", ms=3, color=col, label=f"{name} panel")
    ax[0].plot(alphas, tat, "--", color=col, label=f"{name} thin-airfoil")
    slope = np.polyfit(np.radians(alphas), cl, 1)[0]
    print(f"{name}: dCl/dalpha = {slope:.3f}/rad  (thin-airfoil 2pi = {2*np.pi:.3f}, "
          f"thickness-corrected 2pi(1+0.77t) = {2*np.pi*(1+0.77*t):.3f})")
    print(f"           Cm_c/4 @4deg: panel {solve(*g, 4).cm:+.4f}, thin-airfoil {thin_airfoil(m, p, 4)[1]:+.4f}")
ax[0].set(xlabel="alpha [deg]", ylabel="Cl", title="Lift curve: panel vs thin-airfoil theory")
ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

# 2. Pressure distribution (nodes run TE -> lower -> LE -> upper -> TE)
r = solve(*naca4(0.02, 0.4, 0.12), 4)
half = len(r.xc) // 2
ax[1].plot(r.xc[half:], r.cp[half:], label="upper surface")
ax[1].plot(r.xc[:half], r.cp[:half], label="lower surface")
ax[1].invert_yaxis()
ax[1].set(xlabel="x/c", ylabel="Cp", title=f"NACA 2412, alpha = 4 deg (Cl = {r.cl:.3f})")
ax[1].legend(); ax[1].grid(alpha=.3)

# 3. Grid convergence
ns = [20, 40, 60, 80, 120, 160, 240, 320]
cls = [solve(*naca4(0.02, 0.4, 0.12, n), 4).cl for n in ns]
err = np.abs(np.array(cls[:-1]) - cls[-1])
ax[2].loglog(ns[:-1], err, "o-")
ax[2].set(xlabel="number of panels", ylabel="|Cl - Cl(N=320)|", title="Grid convergence")
ax[2].grid(alpha=.3, which="both")
print("grid convergence (NACA 2412, 4 deg):")
for n, c in zip(ns, cls):
    print(f"  N={n:4d}  Cl={c:.5f}")

fig.tight_layout()
fig.savefig(FIG / "01_solver_validation.png", dpi=130)
print("saved figures/01_solver_validation.png")
