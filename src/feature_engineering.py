import pandas as pd
import re
from typing import Optional


def extract_bhk(title: Optional[str]) -> Optional[int]:
    if pd.isna(title):
        return None
    match = re.search(r'(\d+)\s*BHK', str(title), re.IGNORECASE)
    return int(match.group(1)) if match else None


def luxury_flag(text: Optional[str]) -> int:
    if pd.isna(text):
        return 0
    text = str(text).lower()
    keywords = [
        'premium', 'luxury', 'elite', 'grand', 'sky', 'penthouse',
        'exquisite', 'immaculate', 'pristine', 'stunning'
    ]
    return 1 if any(key in text for key in keywords) else 0


def furnishing_score(value: Optional[str]) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip().lower()
    if text == 'furnished':
        return 2
    if text == 'semi-furnished' or text == 'semi':
        return 1
    return 0


def has_garden_view(overlooking: Optional[str]) -> int:
    if pd.isna(overlooking):
        return 0
    text = str(overlooking).lower()
    return 1 if 'garden' in text or 'park' in text else 0
