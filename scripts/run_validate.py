"""Run data validation checks and write the validation report."""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.ingest import load_config, load_raw_data
from src.data.validate import run_all_validations

config = load_config()
df = load_raw_data(config["data"]["raw_path"])
report = run_all_validations(df)

with open("data/validation_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("Validation:", "PASS" if report["overall_valid"] else "FAIL")
