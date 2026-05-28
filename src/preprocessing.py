import re
import pandas as pd
from typing import Optional, Tuple


def parse_amount(text: Optional[str]) -> Optional[float]:
    if pd.isna(text):
        return None
    text = str(text).strip()
    if text == "":
        return None
    text = text.replace(',', '').replace('₹', '').strip()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)", text)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if 'cr' in unit:
        return value * 10_000_000
    if 'lac' in unit or 'lakh' in unit:
        return value * 100_000
    return None


def parse_area(text: Optional[str]) -> Optional[float]:
    if pd.isna(text):
        return None
    text = str(text).strip()
    if text == "":
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else None


def parse_floor(text: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if pd.isna(text):
        return None, None
    text = str(text).strip().lower()
    if text == "":
        return None, None
    if 'ground' in text:
        match = re.search(r'out of\s*(\d+)', text)
        return 0, int(match.group(1)) if match else None
    match = re.search(r'(\d+)\s*out of\s*(\d+)', text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def parse_parking(text: Optional[str]) -> Tuple[int, str]:
    if pd.isna(text):
        return 0, 'None'
    text = str(text).strip()
    if text == "":
        return 0, 'None'
    match = re.search(r'(\d+)\s*([A-Za-z]+)', text)
    if not match:
        return 0, 'None'
    count = int(match.group(1))
    ptype = match.group(2).capitalize()
    return count, ptype


def extract_bhk(title: Optional[str]) -> Optional[int]:
    if pd.isna(title):
        return None
    match = re.search(r'(\d+)\s*BHK', str(title), re.IGNORECASE)
    return int(match.group(1)) if match else None


def remove_outliers_iqr(df: pd.DataFrame, col: str) -> pd.DataFrame:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)].copy()
