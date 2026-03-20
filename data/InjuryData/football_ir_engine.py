"""
football_ir_engine.py

Production-oriented football injury-risk decision support.

Key capabilities:
- Combines player feedback (1-10), trainer assessments, and objective readiness/workload metrics.
- Supports position-specific stress profiles (GK, DEF, MID, WING, STRIKER).
- Returns calibrated IR chance (%), recommendation, minutes cap, and predicted safe RTP date.
- Enforces mandatory medical override rules (pain, illness, concussion, acute red flags).
- Provides human-readable explainability (top risk drivers + contribution table).

This module is decision support only and does not replace licensed medical judgment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Tuple


POSITION_LOAD_MULTIPLIER = {
    "GK": 0.75,
    "DEF": 1.00,
    "MID": 1.10,
    "WING": 1.18,
    "STRIKER": 1.08,
}


@dataclass
class PlayerFeedback:
    sleep_quality: float            # 1 poor -> 10 excellent
    energy_level: float             # 1 exhausted -> 10 energetic
    stress_level: float             # 1 calm -> 10 high stress (worse)
    soreness_level: float           # 1 none -> 10 severe (worse)
    pain_level: float               # 1 none -> 10 severe (worse)
    confidence_to_train: float      # 1 hesitant -> 10 confident


@dataclass
class TrainerAdvice:
    movement_quality: float         # 1 poor movement -> 10 clean movement
    fatigue_observed: float         # 1 fresh -> 10 very fatigued (worse)
    tissue_red_flags: float         # 1 none -> 10 severe (worse)
    contact_readiness: float        # 1 not ready -> 10 ready
    trainer_influence_weight: float # 0..1


@dataclass
class ObjectiveMetrics:
    hrv_deviation_pct: float        # negative below baseline, ex: -20
    resting_hr_delta_bpm: float     # + bpm over baseline
    acute_load_7d: float
    chronic_load_28d: float
    high_speed_distance_m: float
    days_since_last_injury: int     # 999 if no recent injury
    illness_flag: int               # 1 if ill
    acute_symptom_flag: int = 0     # 1 if acute symptom/significant swelling/limp
    concussion_flag: int = 0        # 1 if concussion protocol active


@dataclass
class PlayerContext:
    player_id: str
    position: str                   # GK / DEF / MID / WING / STRIKER
    age: int
    minutes_last_match: int


@dataclass
class EngineWeights:
    player_section_weight: float = 0.40
    trainer_section_weight: float = 0.30
    objective_section_weight: float = 0.30


class FootballIREngine:
    def __init__(self, weights: EngineWeights | None = None):
        self.weights = weights or EngineWeights()

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def _norm_good(value_1_10: float) -> float:
        # good high -> risk low
        v = FootballIREngine._clamp(value_1_10, 1.0, 10.0)
        return (10.0 - v) / 9.0

    @staticmethod
    def _norm_bad(value_1_10: float) -> float:
        # bad high -> risk high
        v = FootballIREngine._clamp(value_1_10, 1.0, 10.0)
        return (v - 1.0) / 9.0

    @staticmethod
    def _acwr(acute_7d: float, chronic_28d: float) -> float:
        return acute_7d / max(chronic_28d, 1e-6)

    @staticmethod
    def _sigmoid_calibration(raw_risk_0_1: float) -> float:
        # Simple probability calibration curve; tune from historical Brier/reliability results.
        x = FootballIREngine._clamp(raw_risk_0_1, 0.0, 1.0)
        # map around center 0.5 with steeper slope to better separate medium/high risk
        z = (x - 0.5) * 5.0
        return 1.0 / (1.0 + (2.718281828 ** (-z)))

    def _player_risk(self, fb: PlayerFeedback) -> Tuple[float, Dict[str, float]]:
        factors = {
            "sleep": self._norm_good(fb.sleep_quality),
            "energy": self._norm_good(fb.energy_level),
            "stress": self._norm_bad(fb.stress_level),
            "soreness": self._norm_bad(fb.soreness_level),
            "pain": self._norm_bad(fb.pain_level),
            "confidence": self._norm_good(fb.confidence_to_train),
        }
        weights = {
            "sleep": 0.9,
            "energy": 1.0,
            "stress": 0.8,
            "soreness": 1.2,
            "pain": 1.8,
            "confidence": 0.8,
        }
        score = sum(factors[k] * weights[k] for k in factors) / sum(weights.values())
        return score, factors

    def _trainer_risk(self, ta: TrainerAdvice) -> Tuple[float, Dict[str, float]]:
        factors = {
            "movement_quality": self._norm_good(ta.movement_quality),
            "fatigue_observed": self._norm_bad(ta.fatigue_observed),
            "tissue_red_flags": self._norm_bad(ta.tissue_red_flags),
            "contact_readiness": self._norm_good(ta.contact_readiness),
        }
        weights = {
            "movement_quality": 1.1,
            "fatigue_observed": 1.0,
            "tissue_red_flags": 2.0,
            "contact_readiness": 1.0,
        }
        base = sum(factors[k] * weights[k] for k in factors) / sum(weights.values())
        influence = self._clamp(ta.trainer_influence_weight, 0.0, 1.0)
        score = (influence * base) + ((1.0 - influence) * 0.5)
        return score, factors

    def _objective_risk(self, obj: ObjectiveMetrics, ctx: PlayerContext) -> Tuple[float, Dict[str, float]]:
        hrv = self._clamp((-obj.hrv_deviation_pct) / 25.0, 0.0, 1.0)
        rhr = self._clamp((obj.resting_hr_delta_bpm - 2.0) / 10.0, 0.0, 1.0)

        acwr = self._acwr(obj.acute_load_7d, obj.chronic_load_28d)
        if acwr < 0.8:
            load = self._clamp((0.8 - acwr) / 0.5, 0.0, 1.0)
        elif acwr > 1.3:
            load = self._clamp((acwr - 1.3) / 0.7, 0.0, 1.0)
        else:
            load = 0.0

        position_key = ctx.position.upper()
        position_mult = POSITION_LOAD_MULTIPLIER.get(position_key, 1.0)
        hsd_baseline = 900.0 if position_key == "GK" else 1200.0
        hsd = self._clamp(((obj.high_speed_distance_m / position_mult) - hsd_baseline) / 1200.0, 0.0, 1.0)

        reinjury = 0.0
        if obj.days_since_last_injury < 56:
            reinjury = self._clamp((56 - obj.days_since_last_injury) / 56.0, 0.0, 1.0)

        age_load = self._clamp((ctx.age - 30) / 10.0, 0.0, 1.0)
        minute_fatigue = self._clamp((ctx.minutes_last_match - 75) / 30.0, 0.0, 1.0)
        illness = 1.0 if obj.illness_flag else 0.0

        factors = {
            "hrv": hrv,
            "rhr": rhr,
            "load": load,
            "high_speed": hsd,
            "reinjury_window": reinjury,
            "age_modifier": age_load,
            "minutes_last_match": minute_fatigue,
            "illness": illness,
        }
        weights = {
            "hrv": 1.1,
            "rhr": 1.0,
            "load": 1.4,
            "high_speed": 1.0,
            "reinjury_window": 1.4,
            "age_modifier": 0.5,
            "minutes_last_match": 0.7,
            "illness": 2.0,
        }
        score = sum(factors[k] * weights[k] for k in factors) / sum(weights.values())
        return score, factors

    def _override_rules(self, fb: PlayerFeedback, ta: TrainerAdvice, obj: ObjectiveMetrics) -> List[str]:
        rules: List[str] = []
        if obj.concussion_flag == 1:
            rules.append("Concussion protocol active")
        if obj.illness_flag == 1:
            rules.append("Illness / systemic symptoms")
        if obj.acute_symptom_flag == 1:
            rules.append("Acute symptom flag from medical team")
        if fb.pain_level >= 8:
            rules.append("Player pain >= 8/10")
        if ta.tissue_red_flags >= 8:
            rules.append("Trainer tissue red flags >= 8/10")
        return rules

    @staticmethod
    def _risk_band(ir_percent: float) -> str:
        if ir_percent >= 80:
            return "RED"
        if ir_percent >= 65:
            return "ORANGE"
        if ir_percent >= 45:
            return "YELLOW"
        return "GREEN"

    @staticmethod
    def _minutes_cap(ir_percent: float, position: str) -> int:
        base = 90
        if ir_percent >= 80:
            cap = 0
        elif ir_percent >= 65:
            cap = 20
        elif ir_percent >= 45:
            cap = 45
        else:
            cap = 90

        # extra caution for high sprint roles
        if position.upper() in {"WING", "STRIKER"} and cap > 0:
            cap = max(0, cap - 10)
        return cap

    @staticmethod
    def _weeks_out(ir_percent: float, override: bool) -> float:
        if override:
            return 2.0 if ir_percent < 85 else 4.0
        if ir_percent >= 80:
            return 2.0
        if ir_percent >= 65:
            return 1.0
        if ir_percent >= 45:
            return 0.3
        return 0.0

    def estimate(
        self,
        feedback: PlayerFeedback,
        trainer: TrainerAdvice,
        objective: ObjectiveMetrics,
        context: PlayerContext,
        today: date | None = None,
    ) -> Dict[str, object]:
        p_score, p_factors = self._player_risk(feedback)
        t_score, t_factors = self._trainer_risk(trainer)
        o_score, o_factors = self._objective_risk(objective, context)

        w = self.weights
        w_sum = max(w.player_section_weight + w.trainer_section_weight + w.objective_section_weight, 1e-9)
        p_w = w.player_section_weight / w_sum
        t_w = w.trainer_section_weight / w_sum
        o_w = w.objective_section_weight / w_sum

        raw = (p_score * p_w) + (t_score * t_w) + (o_score * o_w)
        calibrated = self._sigmoid_calibration(raw)
        ir_percent = round(calibrated * 100.0, 1)

        overrides = self._override_rules(feedback, trainer, objective)
        force_rest = len(overrides) > 0

        band = self._risk_band(ir_percent)
        if force_rest:
            recommendation = "MUST REST"
        elif band == "RED":
            recommendation = "MUST REST"
        elif band == "ORANGE":
            recommendation = "SHOULD REST"
        elif band == "YELLOW":
            recommendation = "ACTIVE RECOVERY"
        else:
            recommendation = "TRAIN (MONITORED)"

        minutes_cap = self._minutes_cap(ir_percent, context.position)
        weeks_out = self._weeks_out(ir_percent, force_rest)

        base_date = today or date.today()
        safe_rtp_date = base_date + timedelta(days=int(weeks_out * 7))

        # Explainability proxy (feature contribution decomposition)
        contributions = {
            "player_section": round(p_score * p_w * 100.0, 2),
            "trainer_section": round(t_score * t_w * 100.0, 2),
            "objective_section": round(o_score * o_w * 100.0, 2),
        }

        # Top 3 risk drivers across sub-factors
        all_factors: Dict[str, float] = {}
        for k, v in p_factors.items():
            all_factors[f"player.{k}"] = v
        for k, v in t_factors.items():
            all_factors[f"trainer.{k}"] = v
        for k, v in o_factors.items():
            all_factors[f"objective.{k}"] = v

        top_risk_drivers = sorted(all_factors.items(), key=lambda kv: kv[1], reverse=True)[:3]

        return {
            "player_id": context.player_id,
            "position": context.position.upper(),
            "injury_risk_percent": ir_percent,
            "risk_band": band,
            "recommendation": recommendation,
            "minutes_cap": minutes_cap,
            "estimated_weeks_out": weeks_out,
            "predicted_safe_rtp_date": safe_rtp_date.isoformat(),
            "medical_override_rules_triggered": overrides,
            "section_contributions": contributions,
            "top_risk_drivers": top_risk_drivers,
        }


def demo() -> None:
    engine = FootballIREngine()
    result = engine.estimate(
        feedback=PlayerFeedback(
            sleep_quality=4,
            energy_level=5,
            stress_level=7,
            soreness_level=8,
            pain_level=7,
            confidence_to_train=4,
        ),
        trainer=TrainerAdvice(
            movement_quality=5,
            fatigue_observed=7,
            tissue_red_flags=6,
            contact_readiness=4,
            trainer_influence_weight=0.8,
        ),
        objective=ObjectiveMetrics(
            hrv_deviation_pct=-18,
            resting_hr_delta_bpm=7,
            acute_load_7d=3600,
            chronic_load_28d=2400,
            high_speed_distance_m=1450,
            days_since_last_injury=21,
            illness_flag=0,
            acute_symptom_flag=0,
            concussion_flag=0,
        ),
        context=PlayerContext(
            player_id="player_09",
            position="WING",
            age=24,
            minutes_last_match=86,
        ),
    )

    print("=== Sophisticated Injury Risk Decision ===")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    demo()
