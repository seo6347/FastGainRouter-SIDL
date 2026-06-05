from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

ROOT = Path("/data/donggyun/SIDL_project")
BASE = ROOT / "my_experiment" / "skip_restore_v2_tiny"

TABLE_DIR = BASE / "tables"
FEATURE_DIR = BASE / "features"
RESULT_DIR = BASE / "results"
MODEL_DIR = BASE / "models"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_TABLE = TABLE_DIR / "traininner_tiny_base_gain_table_v1.csv"
TUNE_TABLE = TABLE_DIR / "esval_tiny_base_policy_table_v1_lpips.csv"
EVAL_TABLE = TABLE_DIR / "full_val_tiny_base_policy_table_v1_lpips.csv"

TRAIN_FEAT = FEATURE_DIR / "traininner_fast_features_v1.csv"
TUNE_FEAT = FEATURE_DIR / "esval_fast_features_v1.csv"
EVAL_FEAT = FEATURE_DIR / "full_val_fast_features_v1.csv"

MODEL_LATENCY = RESULT_DIR / "model_latency_tiny_base_v2.csv"
FAST_FEATURE_LATENCY = RESULT_DIR / "policyextra_fast_feature_latency_v1.csv"

ACTIONS = ["input", "tiny", "base"]


def load_latency():
    """
    Use robust median latency from repeated runs.
    These are BasicSR test.py end-to-end estimates, not pure forward latency.
    """
    if MODEL_LATENCY.exists():
        lat = pd.read_csv(MODEL_LATENCY)
        tiny_ms = float(lat[lat["branch"] == "tiny"]["ms_per_image"].median())
        base_ms = float(lat[lat["branch"] == "base"]["ms_per_image"].median())
    else:
        tiny_ms = 1.0
        base_ms = 2.0

    if FAST_FEATURE_LATENCY.exists():
        f = pd.read_csv(FAST_FEATURE_LATENCY)
        feature_ms = float(f["feature_ms_per_image"].median())
    else:
        feature_ms = 0.0

    # router predict latency will be measured later
    return tiny_ms, base_ms, feature_ms


TINY_MS, BASE_MS, FEATURE_MS = load_latency()

print("[LATENCY ESTIMATE]")
print("tiny_ms:", TINY_MS)
print("base_ms:", BASE_MS)
print("feature_ms:", FEATURE_MS)


def load_merged(table_path: Path, feature_path: Path, feature_cols=None):
    table = pd.read_csv(table_path)
    feat = pd.read_csv(feature_path)

    df = table.merge(feat, on="input_path", suffixes=("", "_feat"), how="inner")
    if len(df) != len(table):
        print(f"[WARN] merged rows {len(df)} != table rows {len(table)} for {table_path}")

    if feature_cols is None:
        drop_cols = {
            "row_id", "input_path", "filename", "task", "difficulty", "case_key",
            "gt_path", "tiny_path", "base_path",
            "oracle_label_psnr", "oracle_label_ssim", "oracle_label_lpips",
            "split",
        }

        feature_cols = []
        for c in df.columns:
            if c in drop_cols:
                continue
            if c.endswith("_feat"):
                continue
            if c.startswith("psnr_") or c.startswith("ssim_") or c.startswith("lpips_"):
                continue
            if c.startswith("gain_"):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                feature_cols.append(c)

    X = df[feature_cols].copy()
    return df, X, feature_cols


def gain_targets(df):
    return pd.DataFrame({
        "gain_tiny_vs_input": df["gain_tiny_vs_input"],
        "gain_base_vs_tiny": df["gain_base_vs_tiny"],
    })


def predict_policy_from_gain(gain_pred, tau_restore, tau_base):
    pred = []
    for g_tiny, g_base in gain_pred:
        if g_tiny < tau_restore:
            pred.append("input")
        elif g_base > tau_base:
            pred.append("base")
        else:
            pred.append("tiny")
    return np.array(pred, dtype=object)


def selected_arrays(df, pred, include_router=True, router_ms=0.0):
    psnr = np.empty(len(df), dtype=float)
    ssim = np.empty(len(df), dtype=float)
    lpips = np.empty(len(df), dtype=float)
    latency = np.empty(len(df), dtype=float)

    for action in ACTIONS:
        mask = pred == action
        psnr[mask] = df.loc[mask, f"psnr_{action}"].to_numpy()
        ssim[mask] = df.loc[mask, f"ssim_{action}"].to_numpy()
        lpips[mask] = df.loc[mask, f"lpips_{action}"].to_numpy()

        if action == "input":
            action_ms = 0.0
        elif action == "tiny":
            action_ms = TINY_MS
        elif action == "base":
            action_ms = BASE_MS
        else:
            raise ValueError(action)

        # Dynamic policy needs feature extraction and router for every image.
        overhead = FEATURE_MS + router_ms if include_router else 0.0
        latency[mask] = overhead + action_ms

    return psnr, ssim, lpips, latency


def summarize(df, pred, method, include_router=True, router_ms=0.0):
    pred = np.asarray(pred, dtype=object)
    psnr, ssim, lpips, latency = selected_arrays(df, pred, include_router=include_router, router_ms=router_ms)

    return {
        "method": method,
        "psnr": float(psnr.mean()),
        "ssim": float(ssim.mean()),
        "lpips": float(lpips.mean()),
        "latency_ms": float(latency.mean()),
        "input_ratio": float((pred == "input").mean()),
        "tiny_ratio": float((pred == "tiny").mean()),
        "base_ratio": float((pred == "base").mean()),
    }


def save_selected(df, pred, out_path, router_ms=0.0):
    pred = np.asarray(pred, dtype=object)
    psnr, ssim, lpips, latency = selected_arrays(df, pred, include_router=True, router_ms=router_ms)

    out = df.copy()
    out["pred_label"] = pred
    out["selected_psnr"] = psnr
    out["selected_ssim"] = ssim
    out["selected_lpips"] = lpips
    out["selected_latency_ms"] = latency

    selected_path = []
    for _, row in out.iterrows():
        a = row["pred_label"]
        if a == "input":
            selected_path.append(row["input_path"])
        elif a == "tiny":
            selected_path.append(row["tiny_path"])
        elif a == "base":
            selected_path.append(row["base_path"])
        else:
            selected_path.append("")
    out["selected_output_path"] = selected_path

    out.to_csv(out_path, index=False)


def baseline_rows(eval_df):
    rows = []

    # Fixed baselines do not need router/feature overhead.
    for action, name in [
        ("input", "Always Input"),
        ("tiny", "Always Tiny"),
        ("base", "Always Base"),
    ]:
        pred = np.array([action] * len(eval_df), dtype=object)
        rows.append(summarize(eval_df, pred, name, include_router=False, router_ms=0.0))

    # Oracle rows are upper bounds. Cost is calculated as if action were known for free.
    pred_psnr = eval_df["oracle_label_psnr"].astype(str).to_numpy()
    rows.append(summarize(eval_df, pred_psnr, "Oracle PSNR I/T/B", include_router=False, router_ms=0.0))

    pred_lpips = eval_df["oracle_label_lpips"].astype(str).to_numpy()
    rows.append(summarize(eval_df, pred_lpips, "Oracle LPIPS I/T/B", include_router=False, router_ms=0.0))

    return rows


def tune_thresholds(reg, X_tune, tune_df, router_ms):
    gain_tune = reg.predict(X_tune)

    tau_restore_grid = np.round(np.arange(-1.0, 3.01, 0.1), 2)
    tau_base_grid = np.round(np.arange(-1.0, 2.01, 0.1), 2)

    rows = []

    for tr in tau_restore_grid:
        for tb in tau_base_grid:
            pred = predict_policy_from_gain(gain_tune, tr, tb)
            row = summarize(tune_df, pred, "tune", include_router=True, router_ms=router_ms)
            row["tau_restore"] = float(tr)
            row["tau_base"] = float(tb)
            rows.append(row)

    grid = pd.DataFrame(rows)
    grid.to_csv(RESULT_DIR / "fast_gain_router_tuning_grid_traininner_v1.csv", index=False)

    candidates = []

    always_tiny_psnr = float(tune_df["psnr_tiny"].mean())
    always_base_psnr = float(tune_df["psnr_base"].mean())

    # Constraint style: preserve quality, minimize latency.
    for ref_name, ref_psnr in [("tiny", always_tiny_psnr), ("base", always_base_psnr)]:
        for eps in [0.00, 0.03, 0.05, 0.10, 0.20, 0.30]:
            feasible = grid[grid["psnr"] >= ref_psnr - eps].copy()
            if len(feasible) == 0:
                continue

            feasible = feasible.sort_values(
                ["latency_ms", "psnr", "lpips"],
                ascending=[True, False, True],
            )
            best = feasible.iloc[0].to_dict()
            best["criterion"] = f"psnr>={ref_name}-{eps}"
            candidates.append(best)

    # Utility style: PSNR - lambda * normalized latency
    base_latency = BASE_MS
    for lam in [0.00, 0.01, 0.03, 0.05, 0.10, 0.20]:
        tmp = grid.copy()
        tmp["utility"] = tmp["psnr"] - lam * (tmp["latency_ms"] / base_latency)
        best = tmp.sort_values(["utility", "psnr"], ascending=[False, False]).iloc[0].to_dict()
        best["criterion"] = f"utility_lambda={lam}"
        candidates.append(best)

    cand = pd.DataFrame(candidates)
    cand.to_csv(RESULT_DIR / "fast_gain_router_tuning_candidates_traininner_v1.csv", index=False)
    return cand, grid


print("[INFO] load data")
train_df, X_train, feature_cols = load_merged(TRAIN_TABLE, TRAIN_FEAT)
tune_df, X_tune, _ = load_merged(TUNE_TABLE, TUNE_FEAT, feature_cols)
eval_df, X_eval, _ = load_merged(EVAL_TABLE, EVAL_FEAT, feature_cols)

print("[INFO] train rows:", len(train_df))
print("[INFO] tune rows:", len(tune_df))
print("[INFO] eval rows:", len(eval_df))
print("[INFO] features:", feature_cols)

y_train = gain_targets(train_df)

models = {}

ridge = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("reg", Ridge(alpha=1.0)),
])
ridge.fit(X_train, y_train)
models["RidgeGain"] = ridge

rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("reg", MultiOutputRegressor(RandomForestRegressor(
        n_estimators=700,
        max_depth=None,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    ))),
])
rf.fit(X_train, y_train)
models["RFGain"] = rf

summary_rows = baseline_rows(eval_df)
all_candidate_rows = []

for model_name, reg in models.items():
    print()
    print("=" * 80)
    print("[MODEL]", model_name)

    # router predict latency measured on eval
    t0 = time.perf_counter()
    _ = reg.predict(X_eval)
    t1 = time.perf_counter()
    router_ms = (t1 - t0) / len(X_eval) * 1000
    print("[ROUTER] ms/image:", router_ms)

    joblib.dump(
        {"model": reg, "features": feature_cols, "router_ms": router_ms},
        MODEL_DIR / f"fast_gain_router_traininner_{model_name.lower()}_v1.joblib",
    )

    cand, grid = tune_thresholds(reg, X_tune, tune_df, router_ms)
    cand["model"] = model_name
    all_candidate_rows.append(cand)

    gain_eval = reg.predict(X_eval)

    for _, c in cand.iterrows():
        tr = float(c["tau_restore"])
        tb = float(c["tau_base"])
        criterion = c["criterion"]

        pred_eval = predict_policy_from_gain(gain_eval, tr, tb)
        row = summarize(
            eval_df,
            pred_eval,
            method=f"{model_name} {criterion}",
            include_router=True,
            router_ms=router_ms,
        )
        row["model"] = model_name
        row["criterion"] = criterion
        row["tau_restore"] = tr
        row["tau_base"] = tb
        row["router_ms"] = router_ms
        summary_rows.append(row)

        safe = f"{model_name}_{criterion}".replace(">=", "ge").replace("-", "_").replace("=", "_").replace(".", "p").replace(" ", "_").replace("/", "_")
        save_selected(
            eval_df,
            pred_eval,
            RESULT_DIR / f"fast_gain_router_full_val_traininner_{safe}_v1.csv",
            router_ms=router_ms,
        )

summary = pd.DataFrame(summary_rows)
preferred = [
    "method", "psnr", "ssim", "lpips", "latency_ms",
    "input_ratio", "tiny_ratio", "base_ratio",
    "model", "criterion", "tau_restore", "tau_base", "router_ms",
]
cols = [c for c in preferred if c in summary.columns] + [c for c in summary.columns if c not in preferred]
summary = summary[cols]

summary.to_csv(RESULT_DIR / "fast_gain_router_summary_traininner_v1.csv", index=False)

if all_candidate_rows:
    all_cand = pd.concat(all_candidate_rows, ignore_index=True)
    all_cand.to_csv(RESULT_DIR / "fast_gain_router_all_tuning_candidates_traininner_v1.csv", index=False)

print()
print("[FINAL SUMMARY]")
print(summary)

print()
print("[SAVED]")
print(RESULT_DIR / "fast_gain_router_summary_traininner_v1.csv")
print(RESULT_DIR / "fast_gain_router_tuning_grid_traininner_v1.csv")
print(RESULT_DIR / "fast_gain_router_all_tuning_candidates_traininner_v1.csv")
