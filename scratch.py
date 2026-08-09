import json, os
from glob import glob

for path in sorted(glob("results/*metrics*.json")):
    try:
        with open(path) as f:
            data = json.load(f)
        total_models = sum(len(models) for models in data.values())
        print(f"{os.path.basename(path)}: {len(data)} commodities, {total_models} total model runs")
    except Exception as e:
        print(f"Error on {path}: {e}")
