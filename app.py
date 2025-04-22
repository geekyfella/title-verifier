import pandas as pd
import numpy as np
import joblib
import re
import webbrowser
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from metaphone import doublemetaphone
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

# -------------------------------
# Configuration: Common Lists
# -------------------------------
COMMON_PREFIXES = ['THE', 'TODAY', 'JAN', 'CITY', 'RASHTRIYA',
                   'SUNDAY', 'NEW', 'INDIA', 'BHARAT', 'PRADESH', 'LOK', 'INDIAN',
                   'NEWS', 'MAHARASHTRA', 'SHRI', 'DESH', 'HINDUSTAN', 'NAV', 'VIJAY',
                   'NATIONAL', 'MUMBAI', 'DELHI', 'PUBLIC']
COMMON_SUFFIXES = ['TIMES', 'TODAY',   'INDIA', 'TRIBUNE', 'NEWS',
                   'SATURDAY', 'MAGAZINE', 'DAY', 'EXPRESS', 'INTERNATIONAL', 'SUNDAY',
                   'ASCENT', 'KI', 'WEALTH', 'WORLD', 'TOMORROW', 'CITY', 'PLUS',
                   'SAMACHAR', 'PRADESH', 'WEEKLY', 'VOICE']
DISALLOWED_WORDS = [
    "POLICE", "CRIME", "CORRUPTION", "CBI", "CID",
    "ARMY", "NAVY", "TERROR", "TERRORISM",
    "MINISTRY", "GOVERNMENT", "BUREAU"
]
DISALLOWED_PERIODICITY = [
    "DAILY", "WEEKLY", "MONTHLY", "YEARLY", "SUNDAY", "MONDAY",
    "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY",
    "EVENING", "MORNING", "EDITORIAL"
]

# -------------------------------
# FastAPI Setup
# -------------------------------
app = FastAPI(title="Title Verification API")

class TitleRequest(BaseModel):
    title: str

# -------------------------------
# Helper Functions
# -------------------------------
def clean_title(title: str) -> str:
    """Clean a title: convert to uppercase, remove non-alphabetic characters, and extra spaces."""
    if pd.isna(title):
        return ""
    title = title.upper().strip()
    title = re.sub(r'[^A-Z\s]', '', title)
    title = re.sub(r'\s+', ' ', title)
    return title.strip()

def remove_periodicity(title: str, periodicity_words: List[str]) -> str:
    """Remove disallowed periodicity words from the title."""
    tokens = title.split()
    cleaned_tokens = [t for t in tokens if t not in DISALLOWED_PERIODICITY]
    return " ".join(cleaned_tokens)

def check_prefix_suffix(title: str) -> Tuple[List[str], float]:
    """Return warnings if a common prefix or suffix is detected (no penalty applied)."""
    tokens = title.split()
    warning = []
    penalty = 0.0  # Penalty not applied in this version; only warnings are reported.
    if tokens:
        if tokens[0] in COMMON_PREFIXES:
            warning.append(f"Common prefix detected: '{tokens[0]}'")
        if tokens[-1] in COMMON_SUFFIXES:
            warning.append(f"Common suffix detected: '{tokens[-1]}'")
    return warning, penalty

def check_disallowed_words(title: str) -> bool:
    """Return True if the title contains any disallowed word."""
    tokens = title.split()
    return any(token in DISALLOWED_WORDS for token in tokens)

def combination_checker_details(new_title: str, existing_titles: List[str]) -> Tuple[bool, List[str]]:
    """Check if new_title is a combination of words from two or more existing titles.
       Returns (True, [list of matching titles]) if combination is detected."""
    new_tokens = set(new_title.split())
    matching_titles = []
    for ex in existing_titles:
        ex_tokens = set(ex.split())
        if len(new_tokens.intersection(ex_tokens)) / len(new_tokens) >= 0.3:
            matching_titles.append(ex)
    return (len(matching_titles) >= 2, matching_titles)

# -------------------------------
# Similarity Functions using DM and TF-IDF
# -------------------------------
def compute_phonetic_similarity(new_title: str, candidate_title: str) -> float:
    """Compute phonetic similarity using Double Metaphone (DM)."""
    new_clean = clean_title(new_title)
    cand_clean = clean_title(candidate_title)
    dm_new = (doublemetaphone(new_clean))[0]
    dm_cand = (doublemetaphone(cand_clean))[0]
    return fuzz.ratio(dm_new or "", dm_cand or "") / 100.0

def compute_semantic_similarity(new_title: str, candidate_title: str, tfidf_vectorizer: TfidfVectorizer) -> float:
    """Compute semantic similarity using TF-IDF cosine similarity."""
    new_clean = clean_title(new_title)
    cand_clean = clean_title(candidate_title)
    vec_new = tfidf_vectorizer.transform([new_clean])
    vec_cand = tfidf_vectorizer.transform([cand_clean])
    return float(cosine_similarity(vec_new, vec_cand)[0][0])

def compute_verification_probability(new_title: str, existing_titles: List[str],
                                     tfidf_vectorizer: TfidfVectorizer) -> Tuple[float, float, float]:
    """
    Compute overall verification probability from phonetic and semantic similarities.
    Returns (verification probability, max phonetic similarity, max semantic similarity).
    """
    new_clean = clean_title(new_title)
    max_phonetic = 0.0
    max_semantic = 0.0

    for ex in existing_titles:
        p_sim = compute_phonetic_similarity(new_clean, ex)
        s_sim = compute_semantic_similarity(new_clean, ex, tfidf_vectorizer)
        combined = 0.4 * p_sim + 0.6 * s_sim
        if combined > (0.4 * max_phonetic + 0.6 * max_semantic):
            max_phonetic = p_sim
            max_semantic = s_sim

    combined_similarity = 0.4 * max_phonetic + 0.6 * max_semantic
    vp = 1 - combined_similarity
    return vp, max_phonetic, max_semantic

# -------------------------------
# Utility Function: Load Existing Titles
# -------------------------------
def load_existing_titles(filepath: str) -> List[str]:
    """
    Load the existing titles from the CSV file. 
    If the file contains only one column, assume it represents 'cleaned_title'.
    """
    df = pd.read_csv(filepath)
    if "cleaned_title" not in df.columns:
        # If there's only one column, rename it to 'cleaned_title'
        df = df.rename(columns={df.columns[0]: "cleaned_title"})
    df["cleaned_title"] = df["cleaned_title"].astype(str).apply(clean_title)
    return df["cleaned_title"].tolist()

# -------------------------------
# Global Setup: Load TF-IDF Vectorizer and Existing Titles
# -------------------------------
try:
    tfidf_vectorizer = joblib.load("dataset_gen_file/tfidf_vectorizer.pkl")
    print("Loaded saved TF-IDF vectorizer.")
except Exception as e:
    raise Exception("TF-IDF vectorizer is required. Please run the precomputation step.")

EXISTING_TITLES_DB = load_existing_titles("dataset_gen_file/stage1.csv")

# -------------------------------
# API Endpoint Implementation
# -------------------------------
@app.post("/verify-title/")
def verify_title(title_req: TitleRequest):
    new_title_raw = title_req.title
    new_title_clean = clean_title(new_title_raw)

    # Hard reject if disallowed words are present
    if check_disallowed_words(new_title_clean):
        return {
            "status": "rejected",
            "reason": "Title contains disallowed words.",
            "verification_probability": 0.0,
            "passed_phonetic": False,
            "passed_semantic": False,
            "same_title_exists": False,
            "combination_detected": False,
            "combination_titles": []
        }

    # Hard reject if exact title exists
    if new_title_clean in EXISTING_TITLES_DB:
        return {
            "status": "rejected",
            "reason": "Exact title already exists.",
            "verification_probability": 0.0,
            "passed_phonetic": False,
            "passed_semantic": False,
            "same_title_exists": True,
            "combination_detected": False,
            "combination_titles": []
        }

    # Check periodicity: Remove disallowed periodicity words
    new_title_no_period = remove_periodicity(new_title_clean, DISALLOWED_PERIODICITY)
    periodicity_warning = (new_title_clean != new_title_no_period)

    # Check for common prefix/suffix (warnings only)
    prefix_warnings, _ = check_prefix_suffix(new_title_clean)

    # Check if the title is a combination of existing titles
    combination_flag, combination_titles = combination_checker_details(new_title_clean, EXISTING_TITLES_DB)

    # Compute verification probability based on DM and TF-IDF
    vp, max_phonetic, max_semantic = compute_verification_probability(new_title_clean, EXISTING_TITLES_DB, tfidf_vectorizer)

    # Set thresholds for individual checks (example thresholds)
    PHONETIC_THRESHOLD = 0.50
    SEMANTIC_THRESHOLD = 0.50
    passed_phonetic = max_phonetic < PHONETIC_THRESHOLD
    passed_semantic = max_semantic < SEMANTIC_THRESHOLD

    final_vp = vp  # No deductions are applied from warnings
    threshold = 0.60
    if final_vp > threshold:
        status = "accepted"
        message = "Title successfully verified."
    else:
        status = "rejected"
        message = f"Title too similar to existing titles (VP={final_vp:.2f} < {threshold})."

    warnings = []
    if periodicity_warning:
        warnings.append("Periodicity words detected; verification probability may be lower.")
    if prefix_warnings:
        warnings.extend(prefix_warnings)
    if combination_flag:
        warnings.append(f"Title is a combination of existing titles: {', '.join(combination_titles)}.")

    return {
        "status": status,
        "verification_probability": round(final_vp, 2),
        "passed_phonetic": passed_phonetic,
        "passed_semantic": passed_semantic,
        "same_title_exists": False,
        "combination_detected": combination_flag,
        "combination_titles": combination_titles,
        "warnings": warnings,
        "message": message
    }

# -------------------------------
# Auto Open Browser on Startup
# -------------------------------
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8080")

threading.Timer(1.0, open_browser).start()

# -------------------------------
# Main Runner
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)

#uvicorn assembly.main:app --host 127.0.0.1 --port 8080 --reload
