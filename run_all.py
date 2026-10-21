"""Run the whole project from start to finish (about 45 minutes on a normal laptop)."""
import subprocess
import sys
from pathlib import Path

src = Path(__file__).parent / "src"
for script in ["01_data_audit.py", "02_leakage_demo.py", "03_classification.py", "04_regression.py", "05_extensions.py",
                "06_explainability.py", "07_search_report.py"]:
    print(f"\n===== running {script} =====", flush=True)
    subprocess.run([sys.executable, "-W", "ignore", str(src / script)], check=True)
print("\nAll done. See the results/ folder.")
