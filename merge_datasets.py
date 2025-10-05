import argparse
import os
import sys
from typing import Optional, List
import pandas as pd


def debug(msg: str):
    print(f"[merge] {msg}")


def read_csv_robust(path: str) -> pd.DataFrame:
    """Read CSV with fallback encodings and tolerant parsing."""
    encodings = [None, 'utf-8', 'utf-8-sig', 'latin-1', 'ISO-8859-1']
    last_err = None
    for enc in encodings:
        try:
            debug(f"Reading CSV: {path} (encoding={enc})")
            return pd.read_csv(path, encoding=enc, low_memory=False, on_bad_lines='skip')
        except Exception as e:
            last_err = e
            continue
    raise last_err  # type: ignore


def find_candidate_files(dir_path: str, keywords: List[str]) -> List[str]:
    files = []
    for name in os.listdir(dir_path):
        lower = name.lower()
        if lower.endswith('.csv') and any(k in lower for k in keywords):
            files.append(os.path.join(dir_path, name))
    return files


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Identify text column
    text_candidates = [
        'text', 'message', 'email text', 'email_text', 'email', 'body', 'content', 'Email Body', 'Email_Body'
    ]
    label_candidates = [
        'label', 'category', 'class', 'target', 'is_spam', 'spam', 'type'
    ]

    # Normalize columns by lowercasing and stripping spaces/underscores
    rename_map = {col: col.strip().lower().replace('_', ' ') for col in df.columns}
    df = df.rename(columns=rename_map)

    text_col = next((c for c in text_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    # Fallbacks for known schemas
    if text_col is None and 'message' in df.columns:
        text_col = 'message'
    if label_col is None and 'category' in df.columns:
        label_col = 'category'

    if text_col is None:
        raise ValueError(f"Could not find a text column in columns: {list(df.columns)[:10]}...")
    if label_col is None:
        raise ValueError(f"Could not find a label column in columns: {list(df.columns)[:10]}...")

    out = df[[text_col, label_col]].copy()
    out.columns = ['text', 'label']
    return out


def map_labels_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    # If numeric already, coerce to 0/1
    if pd.api.types.is_numeric_dtype(df['label']):
        df['label'] = df['label'].astype(int).clip(0, 1)
        return df

    # Lowercase string labels
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    mapping = {
        # ham/legit
        'ham': 0, 'legit': 0, 'legitimate': 0, 'not spam': 0, 'safe': 0,
        'no': 0, '0': 0, 'false': 0, 'negative': 0,
        # spam/phish
        'spam': 1, 'phish': 1, 'phishing': 1, 'malicious': 1, 'yes': 1,
        '1': 1, 'true': 1, 'positive': 1,
    }
    df['label'] = df['label'].map(lambda x: mapping.get(x, x))

    # Any non-mapped values: attempt heuristic
    df.loc[df['label'].astype(str).str.contains('phish', na=False), 'label'] = 1
    df.loc[df['label'].astype(str).str.contains('spam', na=False), 'label'] = 1
    df.loc[df['label'].astype(str).str.contains('ham|legit', na=False), 'label'] = 0

    # Coerce to numeric, dropping unknowns
    df = df[pd.to_numeric(df['label'], errors='coerce').notna()].copy()
    df['label'] = df['label'].astype(int).clip(0, 1)
    return df


def main():
    parser = argparse.ArgumentParser(description='Merge spam/ham dataset with phishing dataset (mapping phishing→spam).')
    parser.add_argument('--spam', default='spam_dataset.csv', help='Path to the original spam/ham CSV')
    parser.add_argument('--phish', default='', help='Path to the phishing CSV (auto-detect if omitted)')
    parser.add_argument('--out', default='spam_phishing_dataset.csv', help='Output merged CSV path')
    args = parser.parse_args()

    # Load original dataset
    if not os.path.exists(args.spam):
        print(f"ERROR: Could not find spam dataset at {args.spam}")
        sys.exit(1)
    df_orig_raw = read_csv_robust(args.spam)

    # Standardize likely SMS Spam Collection schema
    # Expected columns: Category, Message
    if {'Category', 'Message'}.issubset(set(df_orig_raw.columns)):
        df_orig_raw = df_orig_raw.rename(columns={'Message': 'text', 'Category': 'label'})
    df_orig = standardize_columns(df_orig_raw)
    df_orig = map_labels_to_binary(df_orig)
    debug(f"Original dataset rows after cleaning: {len(df_orig)}")

    # Locate phishing dataset
    phish_path = args.phish
    if not phish_path:
        candidates = find_candidate_files('.', keywords=['phish', 'ceas'])
        # Exclude the main spam dataset if matched by accident
        candidates = [c for c in candidates if os.path.basename(c).lower() != os.path.basename(args.spam).lower()]
        if not candidates:
            print("ERROR: Could not auto-detect a phishing CSV. Pass --phish <path_to_csv>.")
            sys.exit(1)
        # Prefer CEAS-like or explicitly phishing named file
        candidates.sort(key=lambda p: ("phish" not in os.path.basename(p).lower(), os.path.basename(p).lower()))
        phish_path = candidates[0]
        debug(f"Auto-detected phishing dataset: {phish_path}")

    if not os.path.exists(phish_path):
        print(f"ERROR: Phishing CSV not found at {phish_path}")
        sys.exit(1)

    df_phish_raw = read_csv_robust(phish_path)

    # Attempt common renames seen in phishing datasets
    common_renames = {
        'Email Text': 'text', 'Email_Text': 'text', 'EmailText': 'text',
        'Body': 'text', 'Content': 'text', 'Message': 'text',
        'Label': 'label', 'Class': 'label', 'Type': 'label',
    }
    df_phish_raw = df_phish_raw.rename(columns=common_renames)
    df_phish = standardize_columns(df_phish_raw)
    df_phish = map_labels_to_binary(df_phish)
    debug(f"Phishing dataset rows after cleaning: {len(df_phish)}")

    # Keep only text and label, drop NA/dupes
    df_orig = df_orig[['text', 'label']].dropna()
    df_phish = df_phish[['text', 'label']].dropna()

    # Concatenate and clean
    df_merged = pd.concat([df_orig, df_phish], ignore_index=True)
    before = len(df_merged)
    df_merged['text'] = df_merged['text'].astype(str).str.strip()
    df_merged = df_merged[df_merged['text'].str.len() > 0]
    df_merged = df_merged.drop_duplicates(subset=['text'])
    after = len(df_merged)
    debug(f"Merged rows: {before} → {after} after de-duplication and cleanup")

    # Save merged dataset
    df_merged.to_csv(args.out, index=False)
    print(f"Merged dataset saved to {args.out} with {len(df_merged)} rows.")


if __name__ == '__main__':
    main()
