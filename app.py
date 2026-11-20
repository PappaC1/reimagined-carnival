import io
import os
import re
from datetime import datetime
import numpy as np
import pdfplumber
import pandas as pd
import streamlit as st

from racing_model import load_csv, build_features, train_and_predict, choose_selections

st.set_page_config(
    page_title="SA Punters Predictor",
    page_icon="🏇",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("🏇 SA Punters Predictor")
st.caption("Hollywoodbets Punters Challenge • Pure Computaform PDF Engine")

HISTORICAL_FILE = "data/sa_historical_results.csv"
VALUE_EDGE_THRESHOLD = 0.03
MIN_VALUE_BET_ODDS = 2.0


def parse_l3_string(l3_raw: str) -> str:
    """Converts Computaform L3 string (e.g., '449', '0') into '4,4,9'."""
    digits = re.findall(r"\d", str(l3_raw))
    if not digits:
        return ""
    mapped = ["99" if d == "0" else d for d in digits[:3]]
    return ",".join(mapped)


def extract_horse_profiles(page_text: str) -> dict:
    """Extracts FAST FORM ratings and Days Since Last Race/Win per horse."""
    profiles = {}

    fast_block = re.search(r"CFORM RATINGS(.*?)TIPSTER", page_text, re.DOTALL | re.IGNORECASE)
    if fast_block:
        for line in fast_block.group(1).split("\n"):
            if "TOP %" in line or "BEST WEIGHTED" in line:
                continue
            for bm in re.finditer(r"\b(\d{1,2})\s+[A-Za-z][A-Za-z\s\'\(\)\-]+?\s+([+-]\d+\.\d+)\b", line):
                profiles.setdefault(int(bm.group(1)), {})["best_vs_ave"] = float(bm.group(2))
            for im in re.finditer(r"\b(\d{1,2})\s+[A-Za-z][A-Za-z\s\'\(\)\-]+?\s+(\d{1,3})\b", line):
                no = int(im.group(1))
                val = float(im.group(2))
                entry = profiles.setdefault(no, {})
                if "cform" not in entry and val <= 75:
                    entry["cform"] = val
                elif "speed" not in entry:
                    entry["speed"] = val

    horse_sections = re.split(r"\n(?=\d{1,2}\s+[A-Z]{2,})", page_text)
    for sec in horse_sections:
        header_m = re.match(r"^(\d{1,2})\s+[A-Z]{2,}", sec.strip())
        if not header_m:
            continue
        no = int(header_m.group(1))
        entry = profiles.setdefault(no, {})

        dslr_m = re.search(r"(\d+)\s+Days Since Last Race", sec)
        if dslr_m:
            entry["days_since_last_run"] = int(dslr_m.group(1))

        dslw_m = re.search(r"(\d+)\s+Days Since Last Win", sec)
        if dslw_m:
            entry["last_win_days_ago"] = int(dslw_m.group(1))

        entry["recent_dates"] = re.findall(r"\b(\d{2}[A-Za-z]{3}\d{2})\b", sec)

    return profiles


def parse_computaform_pdf(uploaded_file) -> pd.DataFrame:
    """Extracts complete racecard and form features from Computaform PDF."""
    pages_text = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")

    full_text = "\n".join(pages_text)

    course = "FAIRVIEW"
    course_match = re.search(
        r"TAB\s+(FAIRVIEW|TURFFONTEIN|GREYVILLE|KENILWORTH|VAAL|SCOTTSVILLE|DURBANVILLE)",
        full_text,
        re.IGNORECASE,
    )
    if course_match:
        course = course_match.group(1).upper()

    race_date_str = datetime.now().strftime("%Y-%m-%d")
    date_match = re.search(r"(\d{1,2}\s+[A-Z]+\s+20\d{2})", full_text)
    if date_match:
        try:
            race_date_str = pd.to_datetime(date_match.group(1)).strftime("%Y-%m-%d")
        except Exception:
            pass
    race_date_dt = pd.to_datetime(race_date_str)

    going = "Polytrack" if "POLYTRACK" in pages_text.upper() else "Good"

    all_profiles = {}
    current_race_ctx = None

    for txt in pages_text:
        if "Race Ref:" in txt and "WIN - PLACE - SWINGER" in txt:
            r_m = re.search(r"\b\d{2}:\d{2}\s+(\d{1,2})\b", txt)
            if r_m:
                current_race_ctx = int(r_m.group(1))
        elif "Continued" in txt:
            cont_m = re.search(r"RACE\s+(\d{1,2})\s+TAB", txt, re.IGNORECASE)
            if cont_m:
                current_race_ctx = int(cont_m.group(1))

        if current_race_ctx:
            page_prof = extract_horse_profiles(txt)
            for s_no, p_data in page_prof.items():
                all_profiles.setdefault((current_race_ctx, s_no), {}).update(p_data)

    runners = []
    for txt in pages_text:
        if "Race Ref:" not in txt or "WIN - PLACE - SWINGER" not in txt:
            continue

        r_num_match = re.search(r"\b\d{2}:\d{2}\s+(\d{1,2})\b", txt)
        dist_match = re.search(r"\b(\d{3,4})m\b", txt)
        class_match = re.search(r"Computaform Class:\s*([A-Z0-9]+)", txt)

        if not r_num_match:
            continue

        active_race = int(r_num_match.group(1))
        active_distance = int(dist_match.group(1)) if dist_match else 1200
        race_class = class_match.group(1) if class_match else "D"

        if "MAIDEN" in txt.upper():
            active_race_type = "Maiden"
        elif "PLATE" in txt.upper() or "SPRINT" in txt.upper():
            active_race_type = "Plate"
        else:
            active_race_type = "Handicap"

        for line in txt.split("\n"):
            m = re.search(
                r"^\s*(\d{1,2})\s+(?:([\d▲▼]+)\s+)?([A-Z][A-Z\s\'\(\)\-]+?)\s+(\d+)-(\d+)-(\d+)-(\d+)\s+\d+%\s+R[\d,\.]+\s+R[\d,\.]+\s+(\d)([A-Za-z]{1,3})\s+([\d\.]+)(?:-[\d\.]+)?\s+(\d{1,2})\s+([A-Za-z\s/\.\-]+?)\s+(?:[\d\.]+|--)\s*%\s+\*?([A-Za-z\s/\.\-]+?)\s+(?:[\d\.]+|--)\s*%\s*(?:(\d{2,3})\s+(\d{2,3})\s+)?.*?(\d+[-/]\d+)\s*$",
                line,
            )
            if not m:
                continue

            saddle_no = int(m.group(1))
            l3_raw = m.group(2) or ""
            horse_name = m.group(3).strip().title()

            starts = int(m.group(4))
            wins = int(m.group(5))
            seconds = int(m.group(6))
            thirds = int(m.group(7))
            places = wins + seconds + thirds

            age = int(m.group(8))
            acs_tail = m.group(9).lower()
            sex = "G" if acs_tail.endswith("g") else ("F" if acs_tail.endswith(("f", "m")) else "C")

            weight_kg = float(m.group(10))
            draw = int(m.group(11))
            trainer = m.group(12).strip()
            jockey = m.group(13).strip()
            cmr = float(m.group(15)) if m.group(15) else np.nan

            raw_odds = m.group(16).replace("-", "/")
            num, den = map(float, raw_odds.split("/"))
            decimal_odds = round((num / den) + 1.0, 2)

            prof = all_profiles.get((active_race, saddle_no), {})
            runs_90 = 0
            for d_str in prof.get("recent_dates", []):
                try:
                    past_dt = pd.to_datetime(d_str, format="%d%b%y")
                    if 0 <= (race_date_dt - past_dt).days <= 90:
                        runs_90 += 1
                except Exception:
                    pass

            runners.append({
                "race_id": f"{course[:3]}_R{active_race}",
                "race_date": race_date_str,
                "course": course,
                "race_number": active_race,
                "distance_m": active_distance,
                "going": going,
                "class": race_class,
                "race_type": active_race_type,
                "horse_name": horse_name,
                "horse_id": re.sub(r"[^A-Z0-9]", "", horse_name.upper()),
                "trainer_name": trainer,
                "jockey_name": jockey,
                "draw": draw,
                "weight_kg": weight_kg,
                "age": age,
                "sex": sex,
                "days_since_last_run": prof.get("days_since_last_run", np.nan),
                "runs_last_90_days": runs_90,
                "wins_last_12_months": wins,
                "places_last_12_months": places,
                "starts_last_12_months": starts,
                "last_3_positions": parse_l3_string(l3_raw),
                "last_win_days_ago": prof.get("last_win_days_ago", np.nan),
                "official_rating": cmr,
                "market_odds": decimal_odds,
                "finished_position": np.nan,
                "saddle_number": saddle_no,
                "cform_rating": prof.get("cform", np.nan),
                "cf_speed_rating": prof.get("speed", np.nan),
                "best_vs_ave": prof.get("best_vs_ave", np.nan),
            })

    df = pd.DataFrame(runners)
    if not df.empty:
        df = df.drop_duplicates(subset=["race_id", "saddle_number"]).sort_values(
            ["race_number", "saddle_number"]
        )
    return df


def score_without_historical_csv(today_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Computes calibrated win probabilities directly from Computaform PDF features
    when no historical results CSV is present.
    """
    df = today_feat.copy()
    probs = []

    for race_id, group in df.groupby("race_id", sort=False):
        # Normalized market probability (removes overround)
        raw_imp = group["implied_probability"].values
        market_p = raw_imp / raw_imp.sum()

        # Build multi-factor Computaform form & speed index
        cf = group["cform_rating"].fillna(group["cform_rating"].median()).fillna(0).values
        sp = group["cf_speed_rating"].fillna(group["cf_speed_rating"].median()).fillna(0).values
        bva = group["best_vs_ave"].fillna(group["best_vs_ave"].median()).fillna(0).values
        form_sc = group["recent_form_score"].fillna(0).values
        mr = group["official_rating"].fillna(group["official_rating"].median()).fillna(0).values
        win_rt = group["win_rate_12m"].fillna(0).values

        z_score = np.zeros(len(group))
        signals = [
            (cf, 0.30),       # Computaform Rating
            (sp, 0.25),       # CF Speed Rating
            (form_sc, 0.15),  # Last 3 Runs Form Score
            (bva, 0.15),      # Best vs Ave Time This Distance/Course
            (mr, 0.10),       # Current Merit Rating
            (win_rt, 0.05),   # Win Strike Rate
        ]
