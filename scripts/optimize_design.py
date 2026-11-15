"""Step 4: gradient-based airfoil design through the neural surrogate.

Problem (for several target lift coefficients Cl*):
    choose camber m, camber position p, angle of attack alpha   (t fixed at 12%)
    minimise   peak suction  -Cp_min          (flatter pressure, later separation)
    subject to Cl = Cl*                       (required lift)
               Cm_c/4 >= CM_LIMIT             (trim / tail-load limit)

The surrogate gives gradients by backprop to its inputs, so 64 random starts
are optimised with Adam in parallel in a fraction of a second. Each answer is
then checked against the true panel solver, and against a brute-force grid
search carried out entirely with the panel solver.
"""
import time

from _path import ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aeroml.geometry import naca4
from aeroml.mlp import MLP
from aeroml.panel import solve

T_FIXED = 0.12
CM_LIMIT = -0.10
CL_TARGETS = [0.4, 0.6, 0.8, 1.0, 1.2]
LO = np.array([0.00, 0.20, -4.0])   # m, p, alpha
HI = np.array([0.06, 0.60, 12.0])
N_PANELS = 140
FIG = ROOT / "figures"


def surrogate_optimise(net, cl_target, n_starts=64, iters=1500, lr=0.02, w=300.0, w_cm=3e4, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.random((n_starts, 3))                    # design vars scaled to [0, 1]
    m1, m2 = np.zeros_like(u), np.zeros_like(u)

    def full(u):
        z = LO + u * (HI - LO)
        return np.column_stack([z[:, 0], z[:, 1], np.full(len(z), T_FIXED), z[:, 2]])

    for k in range(1, iters + 1):
        X = full(u)
        out = net.predict(X)                          # [Cl, Cm, log(-Cp_min)]
        cl, cm, s = out.T
        viol = np.maximum(CM_LIMIT - cm, 0)
        dJ = np.column_stack([2 * w * (cl - cl_target), -2 * w_cm * viol, np.ones_like(s)])
        gx = net.vjp_input(X, dJ)                     # dJ/d(m, p, t, alpha)
        g = gx[:, [0, 1, 3]] * (HI - LO)
        m1 = 0.9 * m1 + 0.1 * g
        m2 = 0.999 * m2 + 0.001 * g * g
        u -= lr * (m1 / (1 - 0.9**k)) / (np.sqrt(m2 / (1 - 0.999**k)) + 1e-8)
        u = np.clip(u, 0, 1)

    X = full(u)
    cl, cm, s = net.predict(X).T
    J = s + w * (cl - cl_target) ** 2 + w_cm * np.maximum(CM_LIMIT - cm, 0) ** 2
    best = np.argmin(J)
    return X[best], (cl[best], cm[best], -np.exp(s[best]))


def trim_to_cl(m, p, t, cl_target, alpha0=4.0):
    """Find the alpha (deg) at which the panel method gives Cl = cl_target (secant)."""
    g = naca4(m, p, t, N_PANELS)
    a0, a1 = alpha0, alpha0 + 2.0
    c0, c1 = solve(*g, a0).cl, solve(*g, a1).cl
    for _ in range(4):
        a2 = a1 + (cl_target - c1) * (a1 - a0) / (c1 - c0)
        r = solve(*g, a2)
        a0, c0, a1, c1 = a1, c1, a2, r.cl
        if abs(r.cl - cl_target) < 1e-4:
            break
    return a1, r


def grid_ground_truth(cl_target, ms, ps):
    """Brute-force optimum using only the panel solver."""
    cpmin = np.full((len(ps), len(ms)), np.nan)
    cm = np.full_like(cpmin, np.nan)
    for i, p in enumerate(ps):
        for j, m in enumerate(ms):
            _, r = trim_to_cl(m, p, T_FIXED, cl_target)
            cpmin[i, j], cm[i, j] = r.cp_min, r.cm
    feas = np.where(cm >= CM_LIMIT, cpmin, -np.inf)
    i, j = np.unravel_index(np.argmax(feas), feas.shape)
    return (ms[j], ps[i], cpmin[i, j], cm[i, j]), cpmin, cm


def main():
    net = MLP.load(ROOT / "models" / "surrogate.npz")
    ms = np.linspace(0.0, 0.06, 13)
    ps = np.linspace(0.2, 0.6, 9)
    rows, grids = [], {}
    for clt in CL_TARGETS:
        t0 = time.time()
        x, pred = surrogate_optimise(net, clt)
        t_opt = time.time() - t0
        m, p, t, a = x
        r_same = solve(*naca4(m, p, t, N_PANELS), a)             # surrogate's exact design
        a_trim, r_trim = trim_to_cl(m, p, t, clt, a)              # re-trimmed alpha
        t0 = time.time()
        (gm, gp, gcp, gcm), cpg, cmg = grid_ground_truth(clt, ms, ps)
        t_grid = time.time() - t0
        _, r_base = trim_to_cl(0.0, 0.4, T_FIXED, clt)            # NACA 0012 baseline
        grids[clt] = (cpg, cmg, (m, p), (gm, gp))
        rows.append(dict(cl=clt, m=m, p=p, a=a, pred=pred, same=r_same, trim=r_trim,
                         grid=(gm, gp, gcp, gcm), base=r_base, t_opt=t_opt, t_grid=t_grid,
                         n_grid=len(ms) * len(ps)))
        print(f"Cl*={clt:.1f}: surrogate design m={m:.4f} p={p:.3f} alpha={a:.2f}  "
              f"| pred Cl {pred[0]:.3f} Cm {pred[1]:+.3f} Cp_min {pred[2]:.3f}  "
              f"| panel Cl {r_same.cl:.3f} Cm {r_same.cm:+.3f} Cp_min {r_same.cp_min:.3f}  "
              f"| grid best Cp_min {gcp:.3f} (m={gm:.3f}, p={gp:.2f})  "
              f"| NACA0012 Cp_min {r_base.cp_min:.3f}  [{t_opt:.2f}s vs {t_grid:.1f}s]")

    # ---- results table ----
    hdr = ("| Target Cl | Surrogate design (m, p, α) | Predicted Cl / Cm / Cp_min | "
           "Panel check Cl / Cm / Cp_min | Grid-search optimum Cp_min | NACA 0012 Cp_min | "
           "Suction peak reduction vs 0012 |\n|---|---|---|---|---|---|---|")
    body = []
    for r in rows:
        red = 100 * (1 - r["trim"].cp_min / r["base"].cp_min)
        body.append(
            f"| {r['cl']:.1f} | {r['m']:.4f}, {r['p']:.3f}, {r['a']:.2f}° | "
            f"{r['pred'][0]:.3f} / {r['pred'][1]:+.3f} / {r['pred'][2]:.3f} | "
            f"{r['same'].cl:.3f} / {r['same'].cm:+.3f} / {r['same'].cp_min:.3f} | "
            f"{r['grid'][2]:.3f} | {r['base'].cp_min:.3f} | {red:.0f}% |")
    table = hdr + "\n" + "\n".join(body)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "optimisation.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)

    # ---- figure: design landscape, optimum shape, pressure ----
    clt = 0.8
    cpg, cmg, (sm, sp), (gm, gp) = grids[clt]
    r = next(r for r in rows if r["cl"] == clt)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    cs = ax[0].contourf(ms, ps, cpg, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=ax[0], label="Cp_min at Cl = 0.8 (panel method)")
    ax[0].contour(ms, ps, cmg, levels=[CM_LIMIT], colors="r", linewidths=2)
    ax[0].plot([], [], "r-", label=f"Cm = {CM_LIMIT} limit")
    ax[0].plot(gm, gp, "w*", ms=16, mec="k", label="grid-search optimum")
    ax[0].plot(sm, sp, "rx", ms=12, mew=3, label="surrogate optimum")
    ax[0].set(xlabel="max camber m", ylabel="camber position p", title="Design landscape (t = 12%)")
    ax[0].legend(fontsize=8, loc="upper right")

    for (m, p, lab) in [(0, 0.4, "NACA 0012"), (sm, sp, "optimised")]:
        x, y = naca4(m, p, T_FIXED, N_PANELS)
        ax[1].plot(x, y, label=lab)
        _, res = trim_to_cl(m, p, T_FIXED, clt)
        ax[2].plot(res.xc, res.cp, label=f"{lab}  (Cp_min {res.cp_min:.2f}, Cm {res.cm:+.3f})")
    ax[1].set_aspect("equal"); ax[1].set(xlabel="x/c", title="Section shape"); ax[1].legend(loc="lower center", ncol=2, fontsize=8); ax[1].set_ylim(-0.15, 0.15);ax[1].grid(alpha=.3)
    ax[2].invert_yaxis(); ax[2].set(xlabel="x/c", ylabel="Cp", title="Pressure at Cl = 0.8")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "04_optimisation.png", dpi=130)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    cl = [r["cl"] for r in rows]
    ax.plot(cl, [r["base"].cp_min for r in rows], "o-", label="NACA 0012 (baseline)")
    ax.plot(cl, [r["grid"][2] for r in rows], "s-", label="grid search (panel solver)")
    ax.plot(cl, [r["trim"].cp_min for r in rows], "x--", ms=10, mew=2, label="surrogate optimum, panel-verified")
    ax.set(xlabel="target Cl", ylabel="Cp_min (less negative is better)", title="Optimised suction peak vs lift")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "05_design_sweep.png", dpi=130)
    print("saved figures/04_optimisation.png, figures/05_design_sweep.png")


if __name__ == "__main__":
    main()
