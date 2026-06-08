import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)


PROJECT_ROOT = Path(__file__).resolve().parent
PTINFO_PATH = PROJECT_ROOT / "dados" / "EA-MD-QD-2026-02" / "PTdata.xlsx"
FRED_PATH = PROJECT_ROOT / "dados" / "2026-01-MD.csv"
TARGET_COL = "inflation_target"
OUTLIER_PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.pkl"
SCALER_PATH               = PROJECT_ROOT / "models" / "scaler.pkl"


SELECTED_VARIABLES = [
    "HICPNG_PT_ea-md",
    "HICPOV_PT_ea-md",
    "HICPNEF_PT_ea-md",
    "HICPSV_PT_ea-md",
    "GS1_fred-md",
    "GS5_fred-md",
    "OILPRICEx_fred-md_Eur",
    "CUSR0000SAC_fred-md",
    "PCEPI_fred-md",
    "EMPENT_PT_ea-qd",
    "REER42_PT_ea-md",
    "DFGDP_PT_ea-qd",
    "CCONFIX_PT_ea-md",
    "epu_pt_epu",
    "ULCIN_PT_ea-qd",
    "EXPGS_PT_ea-qd",
    "IMPGS_PT_ea-qd",
    "CCI_PT_ea-md",
    "GDP_PT_ea-qd",
    "UNETOT_PT_ea-md",
    "PPIPT_ppi",
]


# Apenas variáveis trimestrais é que necessitam de interpolação
FIXED_INTERPOLATION_METHODS = {
    "EMPENT_PT_ea-qd": "cubic",
    "DFGDP_PT_ea-qd": "spline",
    "ULCIN_PT_ea-qd": "cubic",
    "EXPGS_PT_ea-qd": "cubic",
    "IMPGS_PT_ea-qd": "cubic",
    "GDP_PT_ea-qd": "linear",
}

# Variáveis com ragged edges necessitam do imputaçao implementada com ARIMA
FIXED_ARIMA_MODELS = {
    "CUSR0000SAC_fred-md": (8, 0, 0),
    "EMPENT_PT_ea-qd": (6, 0, 0),
    "DFGDP_PT_ea-qd": (4, 0, 2),
    "ULCIN_PT_ea-qd": (2, 0, 2),
    "EXPGS_PT_ea-qd": (8, 0, 0),
    "IMPGS_PT_ea-qd": (7, 0, 0),
    "GDP_PT_ea-qd": (7, 0, 0),
}

OUTLIER_COLUMNS = [
    'HICPOV_PT_ea-md',
    'HICPNEF_PT_ea-md',
    'HICPSV_PT_ea-md',
    'HICPNG_PT_ea-md',
    'GS5_fred-md',
    'CCONFIX_PT_ea-md',
    'REER42_PT_ea-md',
    'EMPENT_PT_ea-qd',
    'PCEPI_fred-md',
    'EXPGS_PT_ea-qd',
    'IMPGS_PT_ea-qd',
    'CCI_PT_ea-md',
    'UNETOT_PT_ea-md',
    'GS1_fred-md',
    'PPIPT_ppi',
    'ULCIN_PT_ea-qd',
    'GDP_PT_ea-qd',
    'CUSR0000SAC_fred-md',
    'epu_pt_epu',
    'DFGDP_PT_ea-qd',
    'OILPRICEx_fred-md_Eur',
    'inflation_target'
]

SELECTED_LAGS = {
    "GS1_fred-md": [10],
    "GS5_fred-md": [12],
    "OILPRICEx_fred-md_Eur": [1, 2, 3, 4, 5, 6, 7, 12],
    "CUSR0000SAC_fred-md": [1, 2, 11, 12],
    "PCEPI_fred-md": [12],
    "EMPENT_PT_ea-qd": [12],
    "REER42_PT_ea-md": [6, 7, 8, 9, 10, 11, 12],
    "HICPOV_PT_ea-md": [12],
    "HICPNEF_PT_ea-md": [12],
    "HICPSV_PT_ea-md": [12],
    "HICPNG_PT_ea-md": [1, 3, 4, 12],
    "DFGDP_PT_ea-qd": [12],
    "CCONFIX_PT_ea-md": [1, 2, 11, 12],
    "epu_pt_epu": [6, 7, 8],
    "GDP_PT_ea-qd": [1, 2, 3, 4, 5, 12],
    "ULCIN_PT_ea-qd": [12],
    "EXPGS_PT_ea-qd": [12],
    "IMPGS_PT_ea-qd": [2, 3],
    "CCI_PT_ea-md": [12],
    "UNETOT_PT_ea-md": [12],
    "PPIPT_ppi": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
    "inflation_target": [1, 2, 3]
}

FRED_COLUMNS = [
    "INDPRO",
    "IPFINAL",
    "IPCONGD",
    "AWOTMAN",
    "AWHMAN",
    "FEDFUNDS",
    "GS1",
    "GS5",
    "GS10",
    "AAA",
    "BAA",
    "OILPRICEx",
    "CUSR0000SAC",
    "PCEPI",
    "S&P 500",
]


# Lê um dataset, converte Date para índice temporal e remove colunas que não entram na pipeline.
def load_dataset(data):
    if isinstance(data, (str, Path)):
        df = pd.read_csv(data, parse_dates=["Date"])
    else:
        df = data.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

    if "Date" in df.columns:
        df = df.set_index("Date")

    drop_cols = [col for col in ["OILPRICEx_fred-md", "month", "year"] if col in df.columns]
    return df.sort_index().drop(columns=drop_cols)


############################################################################
# SELEÇAO DE VARIAVEIS
######################

# Mantém apenas as variáveis finais decididas para a pipeline.
def select_variables(data, selected_variables=None):
    selected_variables = selected_variables or SELECTED_VARIABLES
    df = load_dataset(data)

    columns_to_keep = list(selected_variables)
    if TARGET_COL in df.columns and TARGET_COL not in columns_to_keep:
        columns_to_keep.append(TARGET_COL)

    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in dataset: {missing_columns}")

    return df[columns_to_keep].copy()

############################################################################



############################################################################
# TRATAMENTO DE MISSING VALUES (INTERPOLAÇÃO + AUTO-ARIMA)
##########################################################

# Aplica o método de interpolação fixo de uma série.
def interpolate_series(series, method):
    kwargs = {
        "method": method,
        "limit": 2,
        "limit_direction": "both",
        "limit_area": "inside",
    }
    if method in {"polynomial", "spline"}:
        kwargs["order"] = 3
    return series.interpolate(**kwargs)


# Aplica os métodos de interpolação já concluídos e devolve um relatório.
def apply_interpolation(df, interpolation_methods=None):
    interpolation_methods = interpolation_methods or FIXED_INTERPOLATION_METHODS
    result = df.copy()
    records = []

    for col in result.columns:
        before = int(result[col].isna().sum())
        method = interpolation_methods.get(col)

        if before == 0:
            records.append({"serie": col, "metodo": None, "status": "no_missing", "n_missing_before": 0, "n_missing_after": 0})
            continue

        if method is None:
            records.append(
                {
                    "serie": col,
                    "metodo": None,
                    "status": "no_fixed_method",
                    "n_missing_before": before,
                    "n_missing_after": before,
                }
            )
            continue

        result[col] = interpolate_series(result[col].astype(float), method)
        after = int(result[col].isna().sum())
        records.append(
            {
                "serie": col,
                "metodo": method,
                "status": "applied",
                "n_missing_before": before,
                "n_missing_after": after,
            }
        )

    report = pd.DataFrame(records).sort_values(["status", "serie"]).reset_index(drop=True)
    return result, report


# Lê os códigos de transformação das variáveis EA e FRED.
def load_transform_specs():
    pt_info = pd.read_excel(PTINFO_PATH, sheet_name="info")[["Name", "TR1", "Frequency"]]
    ea_transform = {}

    for _, row in pt_info.iterrows():
        name = f"{row['Name']}_ea-md" if row["Frequency"] == "M" else f"{row['Name']}_ea-qd"
        ea_transform[name] = int(row["TR1"])

    fred_transform = (
        pd.read_csv(FRED_PATH)[FRED_COLUMNS]
        .rename(columns={col: f"{col}_fred-md" for col in FRED_COLUMNS})
        .iloc[0]
        .to_dict()
    )
    fred_transform["OILPRICEx_fred-md_Eur"] = fred_transform.pop("OILPRICEx_fred-md")
    fred_transform = {key: int(value) for key, value in fred_transform.items()}

    return {"ea": ea_transform, "fred": fred_transform}


# Aplica a transformação de estacionarização das variáveis FRED.
def stacionarize_fred(series, code):
    if code == 1:
        return series
    if code == 2:
        return series.diff()
    if code == 3:
        return series.diff(2)
    if code == 4:
        return np.log(series)
    if code == 5:
        return np.log(series).diff()
    if code == 6:
        return np.log(series).diff(2)
    if code == 7:
        return (series / series.shift() - 1).diff()
    raise ValueError(f"Unsupported FRED transformation code: {code}")


# Aplica a transformação de estacionarização das variáveis EA/PT.
def stacionarize_ea(series, code):
    if code == 1:
        return 100 * stacionarize_fred(series, 4)
    if code == 2:
        return 100 * stacionarize_fred(series, 5)
    if code == 3:
        return 100 * stacionarize_fred(series, 6)
    if code == 4:
        return series
    if code == 5:
        return stacionarize_fred(series, 2)
    raise ValueError(f"Unsupported EA/PT transformation code: {code}")


# Constrói a versão estacionária necessária para prever ragged edges e para a versao final estacionarizada.
def build_stationary_panel(df):
    transform_specs = load_transform_specs()
    stationary = pd.DataFrame(index=df.index)

    for col in df.columns:
        if col == TARGET_COL:
            stationary[col] = stacionarize_fred(df[col], 2)
        elif col == "epu_pt_epu":
            stationary[col] = df[col]
        elif col in transform_specs["fred"]:
            stationary[col] = stacionarize_fred(df[col], transform_specs["fred"][col])
        elif col in transform_specs["ea"]:
            stationary[col] = stacionarize_ea(df[col], transform_specs["ea"][col])
        elif col == "PPIPT_ppi":
            stationary[col] = np.log(df[col] + 6.3)
        else:
            stationary[col] = df[col]

    return stationary.replace([np.inf, -np.inf], np.nan)


# Identifica se a série tem ragged edges e onde estão.
def get_ragged_edges(series):
    valid = series.dropna()
    if valid.empty:
        return None

    first_idx = valid.index[0]
    last_idx = valid.index[-1]
    first_loc = series.index.get_loc(first_idx)
    last_loc = series.index.get_loc(last_idx)
    inner_gap = series.iloc[first_loc : last_loc + 1].isna().any()

    return {
        "first_idx": first_idx,
        "last_idx": last_idx,
        "first_loc": first_loc,
        "last_loc": last_loc,
        "lead_idx": series.index[:first_loc],
        "trail_idx": series.index[last_loc + 1 :],
        "has_internal_nan": bool(inner_gap),
    }


# Diz quantas observações iniciais se perdem após a transformação estacionária.
def get_stationary_offset(transform_group, code):
    if transform_group == "identity":
        return 0
    if transform_group == "ea":
        if code in (1, 4):
            return 0
        if code in (2, 5):
            return 1
        if code == 3:
            return 2
    if transform_group == "fred":
        if code in (1, 4):
            return 0
        if code in (2, 5):
            return 1
        if code in (3, 6, 7):
            return 2
    raise ValueError(f"Unsupported transformation specification: {transform_group=}, {code=}")


# Converte os ragged edges da série original para as posições equivalentes na série estacionária.
def get_stationary_edge_idx(series_index, edge_info, transform_group, code):
    offset = get_stationary_offset(transform_group, code)
    first_loc = edge_info["first_loc"]
    last_loc = edge_info["last_loc"]
    lead_idx = series_index[offset : first_loc + offset] if first_loc > 0 else series_index[:0]
    trail_idx = series_index[last_loc + 1 :] if last_loc < len(series_index) - 1 else series_index[:0]
    return lead_idx, trail_idx


# Reverte as previsões estacionárias para a escala original da série.
def restore_original_scale(series, stationary_filled, transform_group, code, edge_info):
    restored = series.astype(float).copy()
    first_idx, last_idx = edge_info["first_idx"], edge_info["last_idx"]
    first_loc, last_loc = edge_info["first_loc"], edge_info["last_loc"]
    lead_idx = edge_info["lead_idx"]
    trail_idx = edge_info["trail_idx"]

    if transform_group == "identity" or (transform_group == "ea" and code == 4) or (transform_group == "fred" and code == 1):
        idx = lead_idx.union(trail_idx)
        restored.loc[idx] = stationary_filled.loc[idx]
        return restored

    if (transform_group == "ea" and code == 1) or (transform_group == "fred" and code == 4):
        scale = 100.0 if transform_group == "ea" else 1.0
        idx = lead_idx.union(trail_idx)
        restored.loc[idx] = np.exp(stationary_filled.loc[idx] / scale)
        return restored

    if (transform_group == "ea" and code == 2) or (transform_group == "fred" and code == 5):
        scale = 100.0 if transform_group == "ea" else 1.0
        log_levels = pd.Series(index=restored.index, dtype=float)
        observed = restored.loc[first_idx:last_idx]
        log_levels.loc[observed.index] = np.log(observed)

        for pos in range(first_loc, 0, -1):
            curr_idx = restored.index[pos]
            prev_idx = restored.index[pos - 1]
            log_levels.loc[prev_idx] = log_levels.loc[curr_idx] - stationary_filled.loc[curr_idx] / scale

        for pos in range(last_loc + 1, len(restored.index)):
            curr_idx = restored.index[pos]
            prev_idx = restored.index[pos - 1]
            log_levels.loc[curr_idx] = log_levels.loc[prev_idx] + stationary_filled.loc[curr_idx] / scale

        idx = lead_idx.union(trail_idx)
        restored.loc[idx] = np.exp(log_levels.loc[idx])
        return restored

    if (transform_group == "ea" and code == 5) or (transform_group == "fred" and code == 2):
        levels = restored.copy()

        for pos in range(first_loc, 0, -1):
            curr_idx = restored.index[pos]
            prev_idx = restored.index[pos - 1]
            levels.loc[prev_idx] = levels.loc[curr_idx] - stationary_filled.loc[curr_idx]

        for pos in range(last_loc + 1, len(restored.index)):
            curr_idx = restored.index[pos]
            prev_idx = restored.index[pos - 1]
            levels.loc[curr_idx] = levels.loc[prev_idx] + stationary_filled.loc[curr_idx]

        idx = lead_idx.union(trail_idx)
        restored.loc[idx] = levels.loc[idx]
        return restored

    if transform_group == "fred" and code == 3:
        levels = restored.copy()

        for pos in range(first_loc + 1, 1, -1):
            curr_idx = restored.index[pos]
            prev2_idx = restored.index[pos - 2]
            levels.loc[prev2_idx] = levels.loc[curr_idx] - stationary_filled.loc[curr_idx]

        for pos in range(last_loc + 1, len(restored.index)):
            curr_idx = restored.index[pos]
            prev2_idx = restored.index[pos - 2]
            levels.loc[curr_idx] = levels.loc[prev2_idx] + stationary_filled.loc[curr_idx]

        idx = lead_idx.union(trail_idx)
        restored.loc[idx] = levels.loc[idx]
        return restored

    if (transform_group == "ea" and code == 3) or (transform_group == "fred" and code == 6):
        scale = 100.0 if transform_group == "ea" else 1.0
        log_levels = pd.Series(index=restored.index, dtype=float)
        observed = restored.loc[first_idx:last_idx]
        log_levels.loc[observed.index] = np.log(observed)

        for pos in range(first_loc + 1, 1, -1):
            curr_idx = restored.index[pos]
            prev2_idx = restored.index[pos - 2]
            log_levels.loc[prev2_idx] = log_levels.loc[curr_idx] - stationary_filled.loc[curr_idx] / scale

        for pos in range(last_loc + 1, len(restored.index)):
            curr_idx = restored.index[pos]
            prev2_idx = restored.index[pos - 2]
            log_levels.loc[curr_idx] = log_levels.loc[prev2_idx] + stationary_filled.loc[curr_idx] / scale

        idx = lead_idx.union(trail_idx)
        restored.loc[idx] = np.exp(log_levels.loc[idx])
        return restored

    raise ValueError(f"Unsupported restoration rule for {transform_group=}, {code=}")


# Devolve o grupo de transformação e o código usado por cada variável.
def get_transform_spec(column, transform_specs):
    if column == TARGET_COL:
        return {"transform_group": "identity", "code": 1}
    if column == "epu_pt_epu":
        return {"transform_group": "fred", "code": 2}
    if column in transform_specs["fred"]:
        return {"transform_group": "fred", "code": transform_specs["fred"][column]}
    if column in transform_specs["ea"]:
        return {"transform_group": "ea", "code": transform_specs["ea"][column]}
    return {"transform_group": "identity", "code": 1}


# Ajusta o ARIMA fixo decidido em analises feitas para uma série estacionária.
def fit_arima_model(series, order, reverse=False):
    series = pd.Series(series).dropna().astype(float)
    if series.empty:
        raise ValueError("No stationary observations available for ARIMA fitting.")

    if order == (0, 0, 0) or series.nunique() <= 1:
        return {"engine": "constant", "order": order, "constant_value": float(series.iloc[0])}

    y = series.iloc[::-1].reset_index(drop=True) if reverse else series.reset_index(drop=True)
    model = ARIMA(
        y,
        order=order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    return {"engine": "statsmodels", "order": order, "model": model}


# Produz previsões a partir de um modelo ARIMA fixo.
def predict_with_arima(model_info, steps):
    if model_info is None or steps <= 0:
        return np.array([], dtype=float)
    if model_info["engine"] == "constant":
        return np.repeat(model_info["constant_value"], steps).astype(float)
    return np.asarray(model_info["model"].forecast(steps=steps), dtype=float)


# Trata os ragged edges usando um único modelo ARIMA fixo por variável.
def apply_ragged_edge_treatment(df, arima_models=None, strict=True):
    arima_models = arima_models or FIXED_ARIMA_MODELS
    levels = df.copy()
    transform_specs = load_transform_specs()
    stationary = build_stationary_panel(levels)
    records = []

    for col in levels.columns:
        series = levels[col].astype(float)
        edge_info = get_ragged_edges(series)

        if edge_info is None:
            records.append({"serie": col, "status": "no_observations", "arima_order": None})
            continue
        if edge_info["has_internal_nan"]:
            records.append({"serie": col, "status": "internal_missing_remaining", "arima_order": None})
            continue
        if len(edge_info["lead_idx"]) == 0 and len(edge_info["trail_idx"]) == 0:
            records.append({"serie": col, "status": "no_ragged_edges", "arima_order": None})
            continue

        order = arima_models.get(col)
        if order is None:
            if strict:
                raise ValueError(f"Missing fixed ARIMA order for variable with ragged edges: {col}")
            records.append({"serie": col, "status": "no_fixed_arima_model", "arima_order": None})
            continue

        spec = get_transform_spec(col, transform_specs)
        lead_idx, trail_idx = get_stationary_edge_idx(series.index, edge_info, spec["transform_group"], spec["code"])
        stationary_filled = stationary[col].copy()

        if len(lead_idx) > 0:
            lead_model = fit_arima_model(stationary[col], order, reverse=True)
            stationary_filled.loc[lead_idx] = predict_with_arima(lead_model, len(lead_idx))[::-1]

        if len(trail_idx) > 0:
            trail_model = fit_arima_model(stationary[col], order, reverse=False)
            stationary_filled.loc[trail_idx] = predict_with_arima(trail_model, len(trail_idx))

        levels[col] = restore_original_scale(series, stationary_filled, spec["transform_group"], spec["code"], edge_info)
        records.append({"serie": col, "status": "fixed_order_applied", "arima_order": order})

    report = pd.DataFrame(records).sort_values(["status", "serie"]).reset_index(drop=True)
    return levels, report

############################################################################


############################################################################
# TRATAMANETO DE OUTLIERS
#########################

# Monta o dataset em nível esperado pelo preprocessor de outliers.
def build_outlier_input(data, prepared_levels):
    raw = load_dataset(data)
    missing_columns = [col for col in OUTLIER_COLUMNS if col not in raw.columns and col not in prepared_levels.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns for outlier treatment: {missing_columns}")

    df_out = raw.reindex(columns=OUTLIER_COLUMNS).copy()
    for col in prepared_levels.columns:
        if col in df_out.columns:
            df_out[col] = prepared_levels[col]

    return df_out


# Aplica o transformer de outliers ao dataset em nível preparado.
def apply_outlier_treatment(data, prepared_levels):
    df_out = build_outlier_input(data, prepared_levels)
    transformer = joblib.load(OUTLIER_PREPROCESSOR_PATH)
    transformed = transformer.transform(df_out)
    output_columns = [name.split("__", 1)[-1] for name in transformer.get_feature_names_out()]
    return pd.DataFrame(transformed, index=df_out.index, columns=output_columns)

############################################################################


############################################################################
# SELECAO DE LAGS
#################

def validate_selected_lags(selected_variables=None, target_col=TARGET_COL):
    selected_variables = list(selected_variables or SELECTED_VARIABLES)

    variables_to_check = selected_variables.copy()
    if target_col in SELECTED_LAGS and target_col not in variables_to_check:
        variables_to_check.append(target_col)


    missing_lag_spec = [col for col in variables_to_check if col not in SELECTED_LAGS]
    if missing_lag_spec:
        raise KeyError(f"Missing lag specification for selected variables: {missing_lag_spec}")


def apply_lag_selection(df, lag_map=None, target_col=TARGET_COL):
    lag_map = lag_map or SELECTED_LAGS
    lagged_df = pd.DataFrame(index=df.index)
    lag_rows = []

    if target_col in df.columns:
        lagged_df[target_col] = df[target_col]

    for col in df.columns:
        lags = lag_map.get(col, [])
        for lag in lags:
            lagged_df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        lag_rows.append({
            "serie": col,
            "selected_lags": lags,
            "n_lags": len(lags),
        })

    lag_report = pd.DataFrame(lag_rows).sort_values("serie").reset_index(drop=True)
    return lagged_df, lag_report

############################################################################


############################################################################
# STANDARDIZAÇAO DE VARIAVEIS
#############################
def apply_standardization(df, target_col=TARGET_COL, fit_scaler=True):
    result       = df.copy()
    feature_cols = [col for col in result.columns if col != target_col]
    warning      = None

    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        result[feature_cols] = scaler.transform(result[feature_cols])
    elif fit_scaler:
        scaler = StandardScaler()
        result[feature_cols] = scaler.fit_transform(result[feature_cols])
        SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = StandardScaler()
        result[feature_cols] = scaler.fit_transform(result[feature_cols])
        warning = (
            "AVISO: scaler.pkl não encontrado em models/. "
            "A standardização foi aplicada com base nas estatísticas dos dados de input. "
            "Para garantir consistência, processa primeiro os dados de treino."
        )

    return result, scaler, warning


############################################################################


# Prepara um dataset com ou sem seleção de variáveis, interpolação ja definida por variavel, tratamento de ragged edges e opção de estacionarização no output.
def prepare_dataset(data, select_features=True, stationary=False, standardization=False, treat_outliers=False, create_lags=False, fit_scaler=True):
    if stationary and treat_outliers:
        raise ValueError("Outlier treatment is defined only for the level dataset. Use stationary=False when treat_outliers=True.")
    if stationary and not select_features:
        raise ValueError("Stationary output is only supported when select_features=True.")
    if treat_outliers and not select_features:
        raise ValueError("Outlier treatment is only supported when select_features=True.")
    if create_lags and not select_features:
        raise ValueError("Lag creation is only supported when select_features=True.")
    if create_lags and stationary:
        raise ValueError("Lag creation requires stationary=False.")

    selected = select_variables(data) if select_features else load_dataset(data)
    interpolated, interpolation_report = apply_interpolation(selected)
    prepared_levels, ragged_report = apply_ragged_edge_treatment(interpolated, strict=select_features)
    prepared_stationary = build_stationary_panel(prepared_levels)
    prepared_outliers = apply_outlier_treatment(data, prepared_levels) if treat_outliers else None
    prepared_lags = None
    lag_report    = None
    fitted_scaler = None
    std_warning   = None

    if create_lags:
        validate_selected_lags()
        lag_source = prepared_outliers if treat_outliers else prepared_levels
        lagged_only, lag_report = apply_lag_selection(lag_source)
        prepared_lags = pd.concat(
            [lag_source, lagged_only.drop(columns=[TARGET_COL], errors="ignore")],
            axis=1,
        )

    if stationary and not create_lags:
        if standardization:
            prepared, fitted_scaler, std_warning = apply_standardization(prepared_stationary, fit_scaler=fit_scaler)
        else:
            prepared = prepared_stationary
    elif create_lags:
        if standardization:
            prepared, fitted_scaler, std_warning = apply_standardization(prepared_lags, fit_scaler=fit_scaler)
        else:
            prepared = prepared_lags
    elif treat_outliers:
        if standardization:
            prepared, fitted_scaler, std_warning = apply_standardization(prepared_outliers, fit_scaler=fit_scaler)
        else:
            prepared = prepared_outliers
    else:
        if standardization:
            prepared, fitted_scaler, std_warning = apply_standardization(prepared_levels, fit_scaler=fit_scaler)
        else:
            prepared = prepared_levels

    if std_warning:
        print(std_warning)

    return {
        "selected_data": selected,
        "interpolated_data": interpolated,
        "prepared_levels": prepared_levels,
        "prepared_stationary": prepared_stationary,
        "prepared_outliers": prepared_outliers,
        "prepared_lags": prepared_lags,
        "prepared_data": prepared,
        "scaler": fitted_scaler,
        "interpolation_report": interpolation_report,
        "ragged_report": ragged_report,
        "lag_report": lag_report,
    }


# Aplica a mesma pipeline a vários datasets, por exemplo treino e teste.
def prepare_datasets(datasets, select_features=True, stationary=False, treat_outliers=False, create_lags=False):
    return {
        name: prepare_dataset(
            data,
            select_features=select_features,
            stationary=stationary,
            treat_outliers=treat_outliers,
            create_lags=create_lags,
        )
        for name, data in datasets.items()
    }
