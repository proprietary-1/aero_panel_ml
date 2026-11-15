"""Run the whole pipeline end to end (about 10 minutes on a laptop)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for step in ["tests/test_physics.py", "scripts/validate_solver.py", "scripts/generate_data.py",
             "scripts/train_surrogate.py", "scripts/optimize_design.py"]:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, "-u", str(ROOT / step)], check=True, cwd=ROOT)
