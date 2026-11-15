"""Interactive airfoil design app (matplotlib GUI, no extra installs).

    python app.py

Drag the sliders to change the NACA 4-digit shape and angle of attack. The panel
solver recomputes the pressure distribution live, and the neural surrogate's
prediction is shown next to it. "Optimise" runs the gradient-based design search
for the target Cl (t = 12%, Cm >= -0.10, minimum suction peak).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
if "--headless" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from aeroml.geometry import naca4
from aeroml.mlp import MLP
from aeroml.panel import solve
from optimize_design import CM_LIMIT, T_FIXED, surrogate_optimise

N_PANELS = 140
MODEL = ROOT / "models" / "surrogate.npz"
if not MODEL.exists():
    sys.exit("No trained model found. Run:  python run_all.py   (or the scripts in order) first.")
net = MLP.load(MODEL)

fig = plt.figure(figsize=(13, 7.5))
fig.canvas.manager.set_window_title("Airfoil design: panel method + neural surrogate")
ax_shape = fig.add_axes([0.06, 0.62, 0.55, 0.30])
ax_cp = fig.add_axes([0.06, 0.30, 0.55, 0.28])
ax_txt = fig.add_axes([0.66, 0.30, 0.32, 0.62]); ax_txt.axis("off")

sl = {}
specs = [("m", "Max camber m", 0.0, 0.06, 0.02), ("p", "Camber pos. p", 0.2, 0.6, 0.4),
         ("t", "Thickness t", 0.06, 0.18, 0.12), ("a", "Alpha [deg]", -4.0, 12.0, 4.0)]
for i, (key, label, lo, hi, v0) in enumerate(specs):
    sl[key] = Slider(fig.add_axes([0.14, 0.20 - i * 0.04, 0.45, 0.025]), label, lo, hi, valinit=v0)
sl_cl = Slider(fig.add_axes([0.72, 0.20, 0.22, 0.025]), "Target Cl", 0.2, 1.4, valinit=0.8)
btn_opt = Button(fig.add_axes([0.72, 0.12, 0.10, 0.05]), "Optimise")
btn_reset = Button(fig.add_axes([0.84, 0.12, 0.10, 0.05]), "NACA 2412")

(line_shape,) = ax_shape.plot([], [], "C0-", lw=2)
(line_up,) = ax_cp.plot([], [], "C0-", label="upper")
(line_lo,) = ax_cp.plot([], [], "C1-", label="lower")
ax_shape.set(xlim=(-0.02, 1.02), ylim=(-0.15, 0.2), title="Section")
ax_shape.set_aspect("equal", adjustable="box"); ax_shape.grid(alpha=.3)
ax_cp.set(xlim=(-0.02, 1.02), xlabel="x/c", ylabel="Cp"); ax_cp.grid(alpha=.3); ax_cp.legend(loc="lower right")
txt = ax_txt.text(0, 1, "", va="top", family="monospace", fontsize=10.5)
status = {"msg": ""}
busy = {"on": False}


def update(_=None):
    if busy["on"]:
        return
    m, p, t, a = (sl[k].val for k in "mpta")
    x, y = naca4(m, p, t, N_PANELS)
    t0 = time.perf_counter(); r = solve(x, y, a); t_panel = time.perf_counter() - t0
    t0 = time.perf_counter(); cl_n, cm_n, s = net.predict([[m, p, t, a]])[0]; t_nn = time.perf_counter() - t0
    cp_n = -2.718281828 ** s

    half = len(r.xc) // 2
    line_shape.set_data(x, y)
    line_up.set_data(r.xc[half:], r.cp[half:]); line_lo.set_data(r.xc[:half], r.cp[:half])
    ax_cp.set_ylim(max(1.1, r.cp.max() + 0.1), min(-1.0, r.cp.min() * 1.1))  # inverted Cp axis
    name = f"NACA {round(m*100)}{round(p*10) if m >= 0.005 else 0}{round(t*100):02d}"
    ax_shape.set_title(f"{name}-like section   (m={m:.3f}, p={p:.2f}, t={t:.3f}, alpha={a:.1f} deg)")

    trim = "OK" if r.cm >= CM_LIMIT - 5e-4 else "violates trim limit"
    txt.set_text(
        f"{'':10s}{'Panel':>10s}{'Neural net':>12s}\n"
        f"{'-' * 32}\n"
        f"{'Cl':10s}{r.cl:10.3f}{cl_n:12.3f}\n"
        f"{'Cm c/4':10s}{r.cm:+10.4f}{cm_n:+12.4f}\n"
        f"{'Cp_min':10s}{r.cp_min:10.3f}{cp_n:12.3f}\n"
        f"{'time':10s}{t_panel * 1e3:8.1f}ms{t_nn * 1e3:10.2f}ms\n\n"
        f"Cm limit {CM_LIMIT}: {trim}\n\n"
        "Lower |Cp_min| = flatter pressure,\n"
        "gentler adverse gradient, later\n"
        "separation.\n\n"
        f"{status['msg']}")
    fig.canvas.draw_idle()


def set_design(m, p, t, a):
    busy["on"] = True
    for k, v in zip("mpta", (m, p, t, a)):
        sl[k].set_val(v)
    busy["on"] = False
    update()


def on_optimise(_=None):
    clt = sl_cl.val
    t0 = time.perf_counter()
    x, (cl, cm, cp) = surrogate_optimise(net, clt)
    dt = time.perf_counter() - t0
    m, p, t, a = x
    status["msg"] = (f"Optimised for Cl* = {clt:.2f} in {dt:.2f}s\n"
                     f"(64 starts, t fixed at {T_FIXED:.0%},\n"
                     f" Cm >= {CM_LIMIT}). Panel values\n"
                     f" above verify the NN design.")
    set_design(m, p, t, a)


def on_reset(_=None):
    status["msg"] = ""
    set_design(0.02, 0.4, 0.12, 4.0)


for s_ in sl.values():
    s_.on_changed(update)
btn_opt.on_clicked(on_optimise)
btn_reset.on_clicked(on_reset)
update()

if "--headless" in sys.argv:          # used for automated checks / screenshots
    on_optimise()
    out = ROOT / "figures" / "app_screenshot.png"
    fig.savefig(out, dpi=100)
    print(f"headless run OK, saved {out.relative_to(ROOT)}")
else:
    plt.show()
