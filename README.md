# housing-machine-learning

Train and serve a forecasting model for Redfin median sale prices from a CSV or Excel dataset.

## Requirements

- Python 3.12+
- Packages: `pandas`, `scikit-learn`, `statsmodels`, `openpyxl`, `fastapi`, `uvicorn`

## Setup

### Option A: Create your own virtual environment

```powershell
python -m venv housing_venv
.\housing_venv\Scripts\Activate.ps1
pip install pandas scikit-learn statsmodels openpyxl fastapi uvicorn
```

## Input data

- Place your Redfin data file in this folder, or pass a full path.
- Supported formats: `.csv`, `.xlsx`, `.xls`
- The script attempts to auto-detect:
	- target column: `Median Sale Price`
	- month column: `Month of Period End`
	- region column: `Region`

## Run

Train and save the SARIMAX forecasting artifact:

```powershell
python .\housing_prediction.py
```

Start the API locally:

```powershell
python .\api.py
```

## Output

The training script prints chronological holdout metrics and saves a pickle file containing:

- trained per-region SARIMAX models
- detected target column name
- feature column list
- supported regions
- default exogenous values used for future forecasts
