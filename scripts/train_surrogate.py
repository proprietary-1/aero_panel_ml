"""Step 3: train the numpy MLP surrogate and compare against two baselines.

Baselines:
  * thin-airfoil theory (pure physics, no data, no thickness effects)
  * quadratic polynomial regression (classical response surface)
The suction peak is learned as log(-Cp_min), which is far smoother than Cp_min.
"""
import time

from _path import ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aeroml.mlp import MLP
from aeroml.thin_airfoil import thin_airfoil

FIG, MODELS = ROOT / "figures", ROOT / "models"
NAMES = ["Cl", "Cm_c/4", "Cp_min"]
ARCH = [4, 64, 64, 3]


def to_target(Y):
    return np.column_stack([Y[:, 0], Y[:, 1], np.log(-Y[:, 2])])


def from_target(T):
    return np.column_stack([T[:, 0], T[:, 1], -np.exp(T[:, 2])])


def metrics(y, yhat):
    rmse = np.sqrt(np.mean((y - yhat) ** 2))
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return rmse, r2


def quad_features(X):
    Z = (X - X.mean(0)) / X.std(0)
    cols = [np.ones(len(Z))] + [Z[:, i] for i in range(4)]
    cols += [Z[:, i] * Z[:, j] for i in range(4) for j in range(i, 4)]
    return np.column_stack(cols)


def main():
    d = np.load(ROOT / "data" / "dataset.npz")
    X, Y = d["X"], d["Y"]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    n_tr, n_va = int(0.8 * len(X)), int(0.1 * len(X))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    T = to_target(Y)

    # ---- MLP ----
    print(f"training MLP {ARCH} on {len(tr)} samples")
    t0 = time.time()
    net = MLP(ARCH, seed=0)
    hist = net.fit(X[tr], T[tr], X[va], T[va], epochs=1500, lr=3e-3)
    print(f"  done in {time.time() - t0:.1f}s, {len(hist)} epochs")
    MODELS.mkdir(exist_ok=True)
    net.save(MODELS / "surrogate.npz")
    pred_mlp = from_target(net.predict(X[te]))

    # ---- baselines ----
    pred_tat = np.array([thin_airfoil(m, p, a) for m, p, _, a in X[te]])
    F = quad_features(np.vstack([X[tr], X[te]]))
    coef, *_ = np.linalg.lstsq(F[:len(tr)], T[tr], rcond=None)
    pred_poly = from_target(F[len(tr):] @ coef)

    # ---- report ----
    lines = ["| Output | Thin-airfoil theory | Quadratic regression | Neural surrogate |",
             "|---|---|---|---|"]
    for k, name in enumerate(NAMES):
        cells = []
        for pred in (pred_tat if k < 2 else None, pred_poly, pred_mlp):
            if pred is None:
                cells.append("n/a")
            else:
                rmse, r2 = metrics(Y[te, k], pred[:, k])
                cells.append(f"RMSE {rmse:.4f}, R² {r2:.4f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print("\nTest-set accuracy (held-out designs):\n" + table)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "surrogate_metrics.md").write_text(table + "\n", encoding="utf-8")

    # ---- figures: parity plots + training curve ----
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    for k, name in enumerate(NAMES):
        lo, hi = Y[te, k].min(), Y[te, k].max()
        ax[k].plot([lo, hi], [lo, hi], "k-", lw=0.8)
        if k < 2:
            ax[k].scatter(Y[te, k], pred_tat[:, k], s=5, alpha=.4, c="C0", label="thin-airfoil")
        ax[k].scatter(Y[te, k], pred_poly[:, k], s=5, alpha=.4, c="C1", label="quadratic")
        ax[k].scatter(Y[te, k], pred_mlp[:, k], s=5, alpha=.7, c="C2", label="neural net")
        ax[k].set(xlabel=f"{name} (panel method)", ylabel=f"{name} (predicted)", title=name)
        ax[k].legend(fontsize=8); ax[k].grid(alpha=.3)
    ax[3].semilogy(hist[:, 0], label="train"); ax[3].semilogy(hist[:, 1], label="validation")
    ax[3].set(xlabel="epoch", ylabel="MSE (standardised)", title="Training curve")
    ax[3].legend(); ax[3].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "02_surrogate_parity.png", dpi=130)

    # ---- data efficiency: how many solver runs does the surrogate need? ----
    sizes = [100, 250, 500, 1000, 2000, len(tr)]
    errs = []
    for n in sizes:
        sub = net.__class__(ARCH, seed=1)
        sub.fit(X[tr[:n]], T[tr[:n]], X[va], T[va], epochs=800, lr=3e-3, verbose=False)
        p = from_target(sub.predict(X[te]))
        errs.append([metrics(Y[te, k], p[:, k])[0] for k in range(3)])
        print(f"  n_train={n:5d}  RMSE Cl {errs[-1][0]:.4f}  Cm {errs[-1][1]:.4f}  Cp_min {errs[-1][2]:.4f}")
    errs = np.array(errs)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for k, name in enumerate(NAMES):
        ax.loglog(sizes, errs[:, k], "o-", label=name)
    ax.set(xlabel="training samples (panel solves)", ylabel="test RMSE", title="Data efficiency")
    ax.legend(); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(FIG / "03_data_efficiency.png", dpi=130)
    print("saved figures/02_surrogate_parity.png, figures/03_data_efficiency.png")


if __name__ == "__main__":
    main()
