"""Run validation, the complete analysis, figure export, and reference comparison."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
env = os.environ.copy()
cache = ROOT / "outputs" / "cache"
cache.mkdir(parents=True, exist_ok=True)
env["MPLCONFIGDIR"] = str(cache / "matplotlib")
env["XDG_CACHE_HOME"] = str(cache)
env["PYTHONDONTWRITEBYTECODE"] = "1"
for script in ["src/validate_data.py", "src/run_analysis.py", "src/render_figures.py", "src/verify_results.py"]:
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=env, check=True)
print("Analysis, figures, and verification completed.")
