# Project Restructuring & Organization Plan

Since we are keeping both the Commodities and Crypto engines in the exact same repository, we need to transition the repo from a "single-project" layout into a clean, modular "multi-asset" architecture.

If we don't do this now, the root directory will have 50+ python scripts and hundreds of files mixed together, making it impossible to maintain.

Here is the proposed target architecture and the exact steps to achieve it.

---

## 1. The Target Directory Structure

The root directory should only contain the absolute core files (`app.py`, `.env`, `requirements.txt`) and high-level folders.

```text
commodity-price-forecasting/
│
├── app.py                        # The main Streamlit dashboard (will handle both assets eventually)
├── requirements.txt              # Shared Python dependencies
├── .env                          # API Keys
├── project_overview.md           # Master documentation
│
├── data/                         # 🗂️ ALL DATASETS
│   ├── commodities/              # -> Move all current commodity CSVs here
│   ├── crypto/                   # -> Place future crypto CSVs here
│   └── macro-data/               # -> Move current macro data here
│
├── models/                       # 🧠 ALL TRAINED MODELS
│   ├── commodities/              # -> Move `models/`, `models_champion_backup/`, and `models_production/` here
│   └── crypto/                   # -> Future cluster models go here
│
├── results/                      # 📊 METRICS & VALIDATION
│   ├── commodities/              # -> Move all current stage2 JSONs here
│   └── crypto/
│
├── docs/                         # 📝 DOCUMENTATION
│   └── # Move future_works.md, full_metrics.md, horizon.md, sentiment_integration_plan.md, etc.
│
├── notebooks/                    # 📓 JUPYTER NOTEBOOKS
│   └── # Move ALL DL_Training_*.ipynb and ML_Training_*.ipynb (16 files) here
│
└── src/                          # 💻 SOURCE CODE (Python Scripts)
    ├── commodities/              # -> Move stage0 to stage6, ensemble_inference.py, live_data_pipeline.py
    ├── crypto/                   # -> The new clean slate for your Crypto scripts!
    ├── generators/               # -> Move all `create_dl_nb_*.py` and `create_ml_nb_*.py` files here
    ├── analysis/                 # -> Move generate_analysis.py, compare_results.py, generate_fi.py here
    └── utils/                    # -> Move patch.py, revert_losers.py, parse_china.py here
```

---

## 2. Files Recommended for Deletion (The Cleanup)

While reviewing the root directory, I noticed several temporary scratch files, leftovers from web scraping, and blank files. 

**I recommend deleting the following files to declutter the workspace:**
1. `test_app.py` (We just used this for debugging the Streamlit UI, no longer needed).
2. `scratch.py` (Seems to be a 200-byte temporary scratchpad).
3. `get_dom.py` (A 600-byte leftover script).
4. `page_source.html` (Leftover raw HTML from an old web scraper run).
5. `ML_Training.txt` (Appears to be a dumped text log).
6. `project.md` (This file is exactly 1 byte in size and is empty).

---

## 3. The Execution Plan

If you approve of this structure, we can execute this in 3 quick steps:
1. **Clean:** I will run the deletion commands for the 6 junk files listed above.
2. **Create & Move:** I will run a series of Bash `mkdir` and `mv` commands to instantly route all the python scripts, notebooks, and models into their respective new folders.
3. **Path Patching:** Because we are moving files, I will need to quickly patch `app.py` and `stage6_backtester.py` so they look for data in `./data/commodities/` instead of `./data/`.

Take a look at the structure above. If it looks good to you, give me the green light and I will execute the entire cleanup process autonomously!
