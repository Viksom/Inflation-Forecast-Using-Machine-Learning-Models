"""
Para cada seed em 0..n_seeds-1, corre exactamente o mesmo pipeline do CC-VAR.ipynb:
  1. Optuna (RMSE CV, training only) para encontrar os melhores params por subconjunto
  2. PCA + VAR ajustados no treino completo
  3. Previsão multi-step no teste
  4. Reversão de escala e cálculo de MAE, RMSE, WAPE

Score por subconjunto = média(MAE, RMSE, WAPE).
Critério de paragem: se durante patience seeds o mínimo do score (entre os 3 subconjuntos)
não melhorar o melhor mínimo global até aí, para.
"""
import sys
import numpy as np
import pandas as pd
import optuna
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.api import VAR

optuna.logging.set_verbosity(optuna.logging.ERROR)

# ── configuração ──────────────────────────────────────────────────────────────
TARGET = "inflation_target"
THEORETICAL = [
    "epu_pt_epu", "ULCIN_PT_ea-qd", "EXPGS_PT_ea-qd", "IMPGS_PT_ea-qd",
    "CCI_PT_ea-md", "GDP_PT_ea-qd", "UNETOT_PT_ea-md", "PPIPT_ppi",
]
COMPUTATIONAL = [
    "GS1_fred-md", "GS5_fred-md", "OILPRICEx_fred-md_Eur",
    "CUSR0000SAC_fred-md", "PCEPI_fred-md", "EMPENT_PT_ea-qd", "REER42_PT_ea-md",
    "HICPOV_PT_ea-md", "HICPNEF_PT_ea-md", "HICPSV_PT_ea-md",
    "HICPNG_PT_ea-md", "DFGDP_PT_ea-qd", "CCONFIX_PT_ea-md",
]
BEST_FEATURES = [
    "PCEPI_fred-md", "HICPOV_PT_ea-md", "HICPNEF_PT_ea-md",
    "HICPNG_PT_ea-md", "HICPSV_PT_ea-md", "PPIPT_ppi",
    "CCI_PT_ea-md", "EXPGS_PT_ea-qd", "epu_pt_epu",
]
SUBCONJUNTOS = [
    ("Teoricas",       THEORETICAL),
    ("Computacionais", COMPUTATIONAL),
    ("Best Features",  BEST_FEATURES),
]

N_TRIALS = 300
N_FOLDS  = 3
HORIZON  = 12


# ── função principal ──────────────────────────────────────────────────────────
def find_best_seed(train, test, orig, subconjuntos=None, n_seeds=500, n_trials=N_TRIALS, patience=15, verbose=True):
    """
    Pesquisa a melhor seed Optuna para cada subconjunto de variáveis.
    Devolve dict: {nome: {"seed": int, "params": dict, "score": float}}.
    """
    subconjuntos = subconjuntos or SUBCONJUNTOS
    anchor       = orig.loc[orig.index <= train.index[-1]].iloc[-1]

    def reverse_diff(series_diff, anchor_value):
        vals      = series_diff.values
        result    = np.empty(len(vals))
        result[0] = anchor_value + vals[0]
        for i in range(1, len(vals)):
            result[i] = result[i - 1] + vals[i]
        return pd.Series(result, index=series_diff.index)

    def optimizar(train_target, train_X, seed):
        n         = len(train_target)
        k_orig    = train_X.shape[1]
        min_train = max(k_orig * 3 + 1, int(n * 0.55))
        fold_step = (n - min_train - HORIZON) // N_FOLDS

        def objective(trial):
            var_exp    = trial.suggest_float("variancia_explicada", 0.50, 0.95)
            trend      = trial.suggest_categorical("trend", ["nc", "c", "ct"])
            pca_est    = PCA(n_components=var_exp)
            pca_est.fit(train_X.iloc[:min_train])
            k_fold_est = 1 + pca_est.n_components_
            max_lag_k  = max(1, min(6, min_train // (3 * k_fold_est)))
            p          = trial.suggest_int("lag", 1, max_lag_k)
            rmses = []
            for i in range(N_FOLDS):
                cutoff  = min_train + i * fold_step
                val_end = min(cutoff + HORIZON, n)
                tr_y    = train_target.iloc[:cutoff]
                val_y   = train_target.iloc[cutoff:val_end]
                tr_X    = train_X.iloc[:cutoff]
                val_X   = train_X.iloc[cutoff:val_end]
                if len(val_y) == 0:
                    continue
                try:
                    pca_fold = PCA(n_components=var_exp)
                    pca_fold.fit(tr_X)
                    cols_f   = [f"PC{j+1}" for j in range(pca_fold.n_components_)]
                    tr_pca   = pd.DataFrame(pca_fold.transform(tr_X),  index=tr_X.index,  columns=cols_f)
                    vl_pca   = pd.DataFrame(pca_fold.transform(val_X), index=val_X.index, columns=cols_f)
                    tr_var   = pd.concat([tr_y, tr_pca], axis=1)
                    vl_var   = pd.concat([val_y, vl_pca], axis=1)
                    if p >= len(tr_var):
                        return float("inf")
                    res = VAR(tr_var).fit(p, trend=trend)
                    if not res.is_stable(verbose=False):
                        return float("inf")
                    fc  = res.forecast(y=tr_var.values[-p:], steps=len(vl_var))
                    y_p = fc[:, tr_var.columns.get_loc(TARGET)]
                    rmses.append(np.sqrt(mean_squared_error(val_y.values, y_p)))
                except Exception:
                    return float("inf")
            return np.mean(rmses) if rmses else float("inf")

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_params

    def avaliar(variaveis, params):
        try:
            var_exp = params["variancia_explicada"]
            lag     = params["lag"]
            trend   = params["trend"]
            pca     = PCA(n_components=var_exp)
            pca.fit(train[variaveis])
            cols_p  = [f"PC{i+1}" for i in range(pca.n_components_)]
            tr_pca  = pd.DataFrame(pca.transform(train[variaveis]), index=train.index, columns=cols_p)
            te_pca  = pd.DataFrame(pca.transform(test[variaveis]),  index=test.index,  columns=cols_p)
            tr_var  = pd.concat([train[[TARGET]], tr_pca], axis=1)
            te_var  = pd.concat([test[[TARGET]],  te_pca], axis=1)
            if lag >= len(tr_var):
                return None
            res = VAR(tr_var).fit(lag, trend=trend)
            if not res.is_stable(verbose=False):
                return None
            fc        = res.forecast(y=tr_var.values[-lag:], steps=len(te_var))
            pred_orig = reverse_diff(pd.Series(fc[:, 0], index=te_var.index), anchor)
            real_orig = reverse_diff(te_var[TARGET], anchor)
            mae   = round(mean_absolute_error(real_orig.values, pred_orig.values), 4)
            rmse  = round(np.sqrt(mean_squared_error(real_orig.values, pred_orig.values)), 4)
            wape  = round(np.sum(np.abs(real_orig.values - pred_orig.values)) / np.sum(np.abs(real_orig.values)), 4)
            score = round((mae + rmse + wape) / 3, 4)
            return mae, rmse, wape, score
        except Exception:
            return None

    if verbose:
        print(f"A testar {n_seeds} seeds (0..{n_seeds-1}) com {n_trials} trials Optuna cada")
        print(f"Score = média(MAE, RMSE, WAPE) por subconjunto\n")
        print("Critério de paragem: se durante 15 seeds o mínimo do score (entre os 3 subconjuntos) não melhorar o melhor mínimo global até aí, para.\n")
        print(f"{'seed':>5} | {'Teoricas score':>14} | {'Computacionais score':>20} | {'Best Features score':>19}")
        print("-" * 67)

    best            = {nome: {"seed": None, "score": float("inf"), "params": None, "metrics": None} for nome, _ in subconjuntos}
    best_score      = float("inf")
    seeds_no_improv = 0

    for seed in range(n_seeds):
        row = {}
        for nome, variaveis in subconjuntos:
            params = optimizar(train[TARGET], train[variaveis], seed)
            result = avaliar(variaveis, params)
            row[nome] = result[3] if result is not None else float("inf")
            if row[nome] < best[nome]["score"]:
                best[nome] = {"seed": seed, "score": row[nome], "params": params, "metrics": result}

        seed_best = min(row[nome] for nome, _ in subconjuntos)
        improved  = seed_best < best_score
        if improved:
            best_score      = seed_best
            seeds_no_improv = 0
        else:
            seeds_no_improv += 1

        if verbose:
            marker = " *" if improved else ""
            print(f"{seed:5d}  {row['Teoricas']:14.4f}  {row['Computacionais']:20.4f}  {row['Best Features']:19.4f}{marker}")
            sys.stdout.flush()

        if seeds_no_improv >= patience:
            if verbose:
                print(f"\nParagem antecipada: {patience} seeds sem melhoria.\n")
            break

    return {nome: {"seed": b["seed"], "params": b["params"], "score": b["score"], "metrics": b["metrics"]}
            for nome, b in best.items()}


# ── execução como script ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from pipeline_datapreparation import prepare_dataset

    N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    train = prepare_dataset("dados/train.csv", select_features=True, stationary=True, standardization=True, treat_outliers=False, create_lags=False, fit_scaler=True)["prepared_data"]
    test  = prepare_dataset("dados/test.csv",  select_features=True, stationary=True, standardization=True, treat_outliers=False, create_lags=False, fit_scaler=False)["prepared_data"].iloc[12:]
    ALL_VARS = sorted({v for _, vs in SUBCONJUNTOS for v in vs})
    cols  = [TARGET] + ALL_VARS
    train = train[cols].apply(pd.to_numeric, errors="coerce").dropna().asfreq("MS")
    test  = test[cols].apply(pd.to_numeric, errors="coerce").dropna().asfreq("MS")

    compact = pd.read_csv("dados/CompactedData.csv")
    compact["Date"] = pd.to_datetime(compact["Date"])
    orig = compact.set_index("Date")[TARGET]

    results = find_best_seed(train, test, orig, n_seeds=N_SEEDS, verbose=True)

    print("\n" + "=" * 60)
    print("MELHOR SEED POR SUBCONJUNTO (score = média MAE+RMSE+WAPE):")
    variaveis_map = {nome: vars_ for nome, vars_ in SUBCONJUNTOS}
    for nome, b in results.items():
        p  = b["params"]
        nc = PCA(n_components=p["variancia_explicada"]).fit(train[variaveis_map[nome]]).n_components_
        mae, rmse, wape, score = b["metrics"]
        print(f"  {nome}: seed={b['seed']}, score={score:.4f}")
        print(f"    MAE={mae:.4f}, RMSE={rmse:.4f}, WAPE={wape:.4f}")
        print(f"    lag={p['lag']}, trend='{p['trend']}', nc={nc}")
