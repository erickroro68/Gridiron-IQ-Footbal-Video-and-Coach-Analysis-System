"""
football_ir_pipeline.py

Version 2 pipeline for production-like usage:
- CSV ingestion for historical injury labeling (`injury_in_7d`)
- Position-specific model training
- Time-aware validation (no random split leakage)
- Probability calibration (isotonic/sigmoid)
- Optional SHAP explanations when available
- Squad daily scoring with coach-friendly traffic-light report

Requires: pandas, numpy, scikit-learn
Optional: xgboost, shap
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import numpy as np
    import pandas as pd
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    HAS_ML_DEPS = True
except Exception:
    HAS_ML_DEPS = False

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

from football_ir_engine import (
    FootballIREngine,
    ObjectiveMetrics,
    PlayerContext,
    PlayerFeedback,
    TrainerAdvice,
)


FEATURE_COLUMNS = [
    "sleep_quality",
    "energy_level",
    "stress_level",
    "soreness_level",
    "pain_level",
    "confidence_to_train",
    "movement_quality",
    "fatigue_observed",
    "tissue_red_flags",
    "contact_readiness",
    "trainer_influence_weight",
    "hrv_deviation_pct",
    "resting_hr_delta_bpm",
    "acute_load_7d",
    "chronic_load_28d",
    "high_speed_distance_m",
    "days_since_last_injury",
    "illness_flag",
    "acute_symptom_flag",
    "concussion_flag",
    "age",
    "minutes_last_match",
]


@dataclass
class ModelPackage:
    model: Pipeline
    features: List[str]
    position: str


def _require_ml_deps() -> None:
    if not HAS_ML_DEPS:
        raise RuntimeError("Missing dependencies: install numpy, pandas, scikit-learn (and optional xgboost/shap).")


def _build_base_estimator() -> object:
    _require_ml_deps()
    if HAS_XGB:
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
    return RandomForestClassifier(n_estimators=300, random_state=42, min_samples_leaf=3)


def _build_pipeline() -> Pipeline:
    _require_ml_deps()
    estimator = _build_base_estimator()
    base = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    return calibrated


def _time_split(df: "pd.DataFrame", time_col: str = "date", train_ratio: float = 0.8) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    ordered = df.sort_values(time_col).reset_index(drop=True)
    cut = max(1, int(len(ordered) * train_ratio))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def train_position_models(data_path: str, out_dir: str) -> Dict[str, Dict[str, float]]:
    _require_ml_deps()
    df = pd.read_csv(data_path)
    required = set(FEATURE_COLUMNS + ["position", "injury_in_7d", "date"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Dict[str, float]] = {}

    for position, grp in df.groupby("position"):
        if len(grp) < 30:
            continue

        train_df, test_df = _time_split(grp)
        if len(test_df) < 5:
            continue

        model = _build_pipeline()
        x_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["injury_in_7d"].astype(int)

        x_test = test_df[FEATURE_COLUMNS]
        y_test = test_df["injury_in_7d"].astype(int)

        model.fit(x_train, y_train)
        proba = model.predict_proba(x_test)[:, 1]

        auc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan")
        brier = float(brier_score_loss(y_test, proba))

        metrics[position] = {"auc": auc, "brier": brier, "n_train": float(len(train_df)), "n_test": float(len(test_df))}

        import joblib

        path = out / f"model_{position}.joblib"
        joblib.dump(ModelPackage(model=model, features=FEATURE_COLUMNS, position=position), path)

    metrics_path = out / "training_metrics.csv"
    pd.DataFrame.from_dict(metrics, orient="index").to_csv(metrics_path)
    return metrics


def _recommendation_from_prob(prob: float, overrides: List[str]) -> str:
    if overrides or prob >= 0.80:
        return "MUST REST"
    if prob >= 0.65:
        return "SHOULD REST"
    if prob >= 0.45:
        return "ACTIVE RECOVERY"
    return "TRAIN (MONITORED)"


def score_daily_squad(input_csv: str, model_dir: str, output_csv: str) -> "pd.DataFrame":
    _require_ml_deps()
    import joblib

    df = pd.read_csv(input_csv)
    engine = FootballIREngine()
    rows: List[Dict[str, object]] = []

    for _, r in df.iterrows():
        pos = str(r["position"])
        model_path = Path(model_dir) / f"model_{pos}.joblib"

        if model_path.exists():
            package: ModelPackage = joblib.load(model_path)
            x = pd.DataFrame([r[package.features].to_dict()])
            prob = float(package.model.predict_proba(x)[0, 1])
        else:
            # fallback to rules engine if position model not available
            decision = engine.estimate(
                feedback=PlayerFeedback(
                    sleep_quality=float(r["sleep_quality"]),
                    energy_level=float(r["energy_level"]),
                    stress_level=float(r["stress_level"]),
                    soreness_level=float(r["soreness_level"]),
                    pain_level=float(r["pain_level"]),
                    confidence_to_train=float(r["confidence_to_train"]),
                ),
                trainer=TrainerAdvice(
                    movement_quality=float(r["movement_quality"]),
                    fatigue_observed=float(r["fatigue_observed"]),
                    tissue_red_flags=float(r["tissue_red_flags"]),
                    contact_readiness=float(r["contact_readiness"]),
                    trainer_influence_weight=float(r.get("trainer_influence_weight", 0.8)),
                ),
                objective=ObjectiveMetrics(
                    hrv_deviation_pct=float(r["hrv_deviation_pct"]),
                    resting_hr_delta_bpm=float(r["resting_hr_delta_bpm"]),
                    acute_load_7d=float(r["acute_load_7d"]),
                    chronic_load_28d=float(r["chronic_load_28d"]),
                    high_speed_distance_m=float(r["high_speed_distance_m"]),
                    days_since_last_injury=int(r["days_since_last_injury"]),
                    illness_flag=int(r["illness_flag"]),
                    acute_symptom_flag=int(r.get("acute_symptom_flag", 0)),
                    concussion_flag=int(r.get("concussion_flag", 0)),
                ),
                context=PlayerContext(
                    player_id=str(r["player_id"]),
                    position=pos,
                    age=int(r["age"]),
                    minutes_last_match=int(r["minutes_last_match"]),
                ),
            )
            prob = float(decision["injury_risk_percent"]) / 100.0

        overrides = []
        if float(r.get("pain_level", 1)) >= 8:
            overrides.append("Pain >= 8")
        if int(r.get("concussion_flag", 0)) == 1:
            overrides.append("Concussion protocol")
        if int(r.get("acute_symptom_flag", 0)) == 1:
            overrides.append("Acute symptom")

        recommendation = _recommendation_from_prob(prob, overrides)
        color = "red" if recommendation == "MUST REST" else "orange" if recommendation == "SHOULD REST" else "yellow" if recommendation == "ACTIVE RECOVERY" else "green"

        rows.append(
            {
                "player_id": r["player_id"],
                "position": pos,
                "injury_risk_percent": round(prob * 100.0, 1),
                "recommendation": recommendation,
                "color": color,
                "override_reasons": "; ".join(overrides),
            }
        )

    scored = pd.DataFrame(rows)
    scored.to_csv(output_csv, index=False)
    return scored


def explain_with_shap(input_csv: str, model_dir: str, player_id: str) -> str:
    _require_ml_deps()
    if not HAS_SHAP:
        return "SHAP not installed. Install shap to generate feature explanations."

    import joblib

    df = pd.read_csv(input_csv)
    row = df.loc[df["player_id"].astype(str) == str(player_id)]
    if row.empty:
        return f"Player {player_id} not found in input CSV."

    row = row.iloc[0]
    pos = str(row["position"])
    model_path = Path(model_dir) / f"model_{pos}.joblib"
    if not model_path.exists():
        return f"No model for position {pos}."

    package: ModelPackage = joblib.load(model_path)
    x = pd.DataFrame([row[package.features].to_dict()])

    estimator = package.model
    explainer = shap.Explainer(estimator.predict_proba, x)
    sv = explainer(x)
    order = np.argsort(np.abs(sv.values[0, :, 1]))[::-1][:5]
    top = [(package.features[i], float(sv.values[0, i, 1])) for i in order]
    return f"Top SHAP features for {player_id}: {top}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Football IR V2 pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--out", required=True)

    score = sub.add_parser("score")
    score.add_argument("--input", required=True)
    score.add_argument("--models", required=True)
    score.add_argument("--out", required=True)

    explain = sub.add_parser("explain")
    explain.add_argument("--input", required=True)
    explain.add_argument("--models", required=True)
    explain.add_argument("--player-id", required=True)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.cmd == "train":
        metrics = train_position_models(args.data, args.out)
        print("Training complete:")
        print(metrics)
    elif args.cmd == "score":
        scored = score_daily_squad(args.input, args.models, args.out)
        print(scored.head().to_string(index=False))
    elif args.cmd == "explain":
        message = explain_with_shap(args.input, args.models, args.player_id)
        print(message)


if __name__ == "__main__":
    main()
