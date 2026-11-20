"""
South African Horse Racing Prediction Model
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "race_id",
    "race_date",
    "course",
    "race_number",
    "distance_m",
    "going",
    "class",
    "race_type",
    "horse_name",
    "horse_id",
    "trainer_name",
    "jockey_name",
    "draw",
    "weight_kg",
    "age",
    "sex",
    "days_since_last_run",
    "runs_last_90_days",
    "wins_last_12_months",
    "places_last_12_months",
    "starts_last_12_months",
    "last_3_positions",
    "last_win_days_ago",
    "official_rating",
    "market_odds",
    "finished_position",
]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def parse_last_3_positions(value) -> list[int]:
    if pd.isna(value) or str(value).strip() == "":
        return [99, 99, 99]

    raw_positions = str(value).replace(" ", "").split(",")
    parsed: list[int] = []

    for raw in raw_positions[:3]:
        try:
            parsed.append(int(float(raw)))
        except ValueError:
            parsed.append(99)

    while len(parsed) < 3:
        parsed.append(99)

    return parsed


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = [
        "race_number",
        "distance_m",
        "draw",
        "weight_kg",
        "age",
        "days_since_last_run",
        "runs_last_90_days",
        "wins_last_12_months",
        "places_last_12_months",
        "starts_last_12_months",
        "last_win_days_ago",
        "official_rating",
        "market_odds",
        "finished_position",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

    positions = df["last_3_positions"].apply(parse_last_3_positions)

    df["last_finish"] = [p[0] for p in positions]
    df["second_last_finish"] = [p[1] for p in positions]
    df["third_last_finish"] = [p[2] for p in positions]

    df["recent_form_score"] = (
        (4 - df["last_finish"].clip(upper=4)) * 3
        + (4 - df["second_last_finish"].clip(upper=4)) * 2
        + (4 - df["third_last_finish"].clip(upper=4))
    )

    df["win_rate_12m"] = df["wins_last_12_months"] / df[
        "starts_last_12_months"
    ].replace(0, np.nan)

    df["place_rate_12m"] = df["places_last_12_months"] / df[
        "starts_last_12_months"
    ].replace(0, np.nan)

    df["market_odds"] = df["market_odds"].clip(lower=1.01)
    df["implied_probability"] = 1.0 / df["market_odds"]
    df["log_odds"] = np.log(df["market_odds"])

    df["market_rank"] = df.groupby("race_id")["market_odds"].rank(method="min")
    df["field_size"] = df.groupby("race_id")["horse_name"].transform("count")

    default_rate = 0.12
    df["course_distance_win_rate"] = default_rate
    df["course_win_rate"] = default_rate
    df["trainer_win_rate"] = default_rate
    df["jockey_win_rate"] = default_rate

    return df


def train_and_predict(*args, **kwargs):
    raise NotImplementedError("Historical CSV mode is disabled.")


@dataclass
class Selection:
    race_id: str
    race_number: int
    course: str
    selection_type: str
    horse_name: str
    model_win_probability: float
    market_odds: float
    model_edge: float
    rank: int
    is_best_selection: bool


def choose_selections(
    predictions: pd.DataFrame,
    value_edge_threshold: float = 0.03,
    min_value_bet_odds: float = 2.0,
) -> list[Selection]:
    selections: list[Selection] = []

    for race_id, race in predictions.groupby("race_id", sort=False):
        race = race.sort_values(
            by=["model_win_probability", "market_odds"],
            ascending=[False, True],
        ).reset_index(drop=True)

        if race.empty:
            continue

        probable_winner = race.iloc[0]

        selections.append(
            Selection(
                race_id=race_id,
                race_number=int(race["race_number"].iloc[0]),
                course=str(race["course"].iloc[0]),
                selection_type="Probable Winner",
                horse_name=str(probable_winner["horse_name"]),
                model_win_probability=float(
                    probable_winner["model_win_probability"]
                ),
                market_odds=float(probable_winner["market_odds"]),
                model_edge=float(probable_winner["model_edge"]),
                rank=1,
                is_best_selection=True,
            )
        )

        value_candidates = race[
            (race["model_edge"] >= value_edge_threshold)
            & (race["market_odds"] >= min_value_bet_odds)
        ].sort_values(by="model_edge", ascending=False)

        value_bet: Optional[pd.Series] = None
        if not value_candidates.empty:
            value_bet = value_candidates.iloc[0]
            selections.append(
                Selection(
                    race_id=race_id,
                    race_number=int(race["race_number"].iloc[0]),
                    course=str(race["course"].iloc[0]),
                    selection_type="Value Bet",
                    horse_name=str(value_bet["horse_name"]),
                    model_win_probability=float(
                        value_bet["model_win_probability"]
                    ),
                    market_odds=float(value_bet["market_odds"]),
                    model_edge=float(value_bet["model_edge"]),
                    rank=2,
                    is_best_selection=False,
                )
            )

        danger_candidates = race.iloc[1:].copy()
        if value_bet is not None:
            danger_candidates = danger_candidates[
                danger_candidates["horse_name"] != value_bet["horse_name"]
            ]

        danger_horse = (
            race.iloc[-1] if danger_candidates.empty else danger_candidates.iloc[0]
        )

        selections.append(
            Selection(
                race_id=race_id,
                race_number=int(race["race_number"].iloc[0]),
                course=str(race["course"].iloc[0]),
                selection_type="Danger Horse",
                horse_name=str(danger_horse["horse_name"]),
                model_win_probability=float(
                    danger_horse["model_win_probability"]
                ),
                market_odds=float(danger_horse["market_odds"]),
                model_edge=float(danger_horse["model_edge"]),
                rank=3,
                is_best_selection=False,
            )
        )

    return selections
