import pandas as pd
import numpy as np

def clean_ids_columns(df_dict):
    """
    Cleans ID columns across all DataFrames by filling NaN/empty values with "0"
    if the corresponding (U) column equals 0.

    Args:
        df_dict: Dictionary of DataFrames {sheet_name: df}

    Returns:
        Updated DataFrames with cleaned ID columns.
    """
    COLUMN_PAIRS = [
        ('Sols coupés (u)', 'Sols coupés (Ids)'),
        ('Sols coupants (u)', 'Sols coupants (Ids)'),
        ('Murs coupés (u)', 'Murs coupés (Ids)'),
        ('Murs coupants (u)', 'Murs coupants (Ids)'),
        ('Poutres coupés (u)', 'Poutres coupés (Ids)'),
        ('Poutres coupants (u)', 'Poutres coupants (Ids)'),
        ('Poteaux coupés (u)', 'Poteaux coupés (Ids)'),
        ('Poteaux coupants (u)', 'Poteaux coupants (Ids)')
    ]

    for sheet_name, df in df_dict.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"⚠️ Skipping {sheet_name}: Empty or invalid DataFrame")
            continue

        print(f"\n🔍 Processing {sheet_name}...")

        for u_col, ids_col in COLUMN_PAIRS:
            if u_col not in df.columns or ids_col not in df.columns:
                print(f"🚨 Skipping: {u_col} or {ids_col} not found in {sheet_name}")
                continue  # Skip missing columns

            print(f"✅ Processing Column Pair: {u_col} → {ids_col}")

            # Convert U column to numeric
            df[u_col] = pd.to_numeric(df[u_col], errors='coerce').fillna(0).astype(int)

            # Convert ID column to **consistent string type**
            df[ids_col] = df[ids_col].astype(str)

            # Explicitly replace known NaN representations with np.nan
            df[ids_col] = df[ids_col].replace(['nan', 'na', 'none', '', ' '], np.nan)

            # Ensure IDs are properly formatted when U = 0
            condition = (df[u_col] == 0) & (df[ids_col].isna())

            df.loc[condition, ids_col] = "0"  # Assigning "0" as string

            cleaned_count = condition.sum()
            print(f"✅ Cleaned {cleaned_count} rows in {ids_col}")

    return df_dict
