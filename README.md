# Inflation Forecast Using Machine Learning Models

This project contains a sequence of Jupyter notebooks and helper Python scripts for preparing data, analyzing time series, and building inflation forecasting models with statistical and machine learning methods.

## Setup

Create and activate a virtual environment before running the notebooks:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies from the project root:

```powershell
python -m pip install -r requirements.txt
```

If you are running the notebooks from VS Code or Jupyter, select the Python interpreter from the `.venv` environment.

## Dependencies

The `requirements.txt` file lists the installable Python packages needed by the notebooks and scripts, including:

- data handling and file formats: `pandas`, `numpy`, `openpyxl`, `pyarrow`, `defusedxml`
- plotting: `matplotlib`, `seaborn`, `plotly`
- time series and statistics: `statsmodels`, `pmdarima`, `scipy`
- machine learning: `scikit-learn`, `lightgbm`
- optimization and model utilities: `optuna`, `joblib`
- notebook support: `ipython`, `ipykernel`, `nbformat`

Some imports used in the notebooks are submodules, not separate packages. For example, `matplotlib.pyplot` and `matplotlib.dates` are installed as part of `matplotlib`, so they should not be added to `requirements.txt` as separate dependencies.

## Running the Notebooks

The notebooks are numbered in the intended workflow order, from data preparation to final model evaluation:

```text
01_DataSourcesPreparation.ipynb
02_DataIntegration.ipynb
03_TimeSeriesAnalysis.ipynb
...
12_FinalModels.ipynb
```

Run them from the project root so that relative paths to data files, saved models, and helper scripts resolve correctly.
