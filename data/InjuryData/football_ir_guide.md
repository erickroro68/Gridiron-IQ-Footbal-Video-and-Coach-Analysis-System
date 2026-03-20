# Football Injury-Risk Guide (Production Version)

## 1) Realistic input scales (1-10)

### Player feedback
- `sleep_quality`: 1 = very poor, 10 = excellent
- `energy_level`: 1 = exhausted, 10 = highly energetic
- `stress_level`: 1 = calm, 10 = highly stressed
- `soreness_level`: 1 = none, 10 = severe
- `pain_level`: 1 = none, 10 = severe
- `confidence_to_train`: 1 = hesitant, 10 = fully confident

### Trainer assessment
- `movement_quality`: 1 = poor asymmetrical movement, 10 = clean movement
- `fatigue_observed`: 1 = fresh, 10 = heavily fatigued
- `tissue_red_flags`: 1 = none, 10 = severe concern
- `contact_readiness`: 1 = not ready, 10 = ready
- `trainer_influence_weight`: 0.0 to 1.0 (how strongly trainer inputs influence score)

### Objective metrics
- `hrv_deviation_pct`: negative values indicate worse than baseline
- `resting_hr_delta_bpm`: positive values indicate elevated stress/fatigue
- `acute_load_7d`, `chronic_load_28d`
- `high_speed_distance_m`
- `days_since_last_injury`
- `illness_flag`, `acute_symptom_flag`, `concussion_flag`

---

## 2) What the engine outputs
- `injury_risk_percent` (calibrated probability)
- `risk_band` (GREEN / YELLOW / ORANGE / RED)
- `recommendation` (TRAIN, ACTIVE RECOVERY, SHOULD REST, MUST REST)
- `minutes_cap` (recommended max match minutes)
- `estimated_weeks_out`
- `predicted_safe_rtp_date`
- `medical_override_rules_triggered`
- explainability (`section_contributions` and `top_risk_drivers`)

---

## 3) Mandatory medical override logic
If any of these are true, recommendation is forced to `MUST REST`:
- Concussion protocol is active
- Acute symptom flag from medical team
- Pain >= 8/10
- Trainer tissue red flags >= 8/10
- Illness/systemic symptoms

---

## 4) Version 2 pipeline commands

### Train position-specific models (time-aware split + calibration)
```bash
python3 football_ir_pipeline.py train --data historical_team_data.csv --out models
```

### Score full squad for the day
```bash
python3 football_ir_pipeline.py score --input daily_squad_inputs.csv --models models --out daily_report.csv
```

### Optional SHAP explainability for one player
```bash
python3 football_ir_pipeline.py explain --input daily_squad_inputs.csv --models models --player-id 9
```

---

## 5) Suggested minimum historical CSV columns
- `date`, `player_id`, `position`, `injury_in_7d`
- all feature columns used in `football_ir_pipeline.py` (`FEATURE_COLUMNS`)

---

## 6) Operational best practices
- Retrain weekly or monthly.
- Track false negatives (missed injuries) aggressively.
- Monitor calibration (Brier score + reliability plots).
- Keep per-position models (`GK`, `DEF`, `MID`, `WING`, `STRIKER`).
- Keep clinician override authority above model output.
