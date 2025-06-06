import pandas as pd
import numpy as np
import os
from bimpredictapp.params import *

from IPython.display import display, HTML
import missingno as msno
import matplotlib.pyplot as plt

def clean_ids_columns(df_dict) -> tuple:
    """
    Compact cleaning with side-by-side before/after comparison and final columns.
    """
    renamed_columns = {}
    print("🛁 Starting ID columns cleaning...\n")

    for sheet_name, df in df_dict.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        u_cols = [col for col in df.columns if ("coupés_u" in col or "coupants_u" in col)]
        if not u_cols:
            continue

        print(f"📋 {sheet_name}:")

        for u_col in u_cols:
            ids_col = u_col.replace('_u', '_Ids')
            if ids_col not in df.columns:
                print(f" ⚠️ {u_col} → No matching IDs column")
                continue

            # Store original values
            original_u = df[u_col].copy()
            original_ids = df[ids_col].copy()

            # Perform cleaning
            df[u_col] = pd.to_numeric(df[u_col], errors='coerce').fillna(0).astype(int)
            df[ids_col] = df[ids_col].astype(str).replace(
                ['nan', 'na', 'none', '', ' '], np.nan
            )
            condition = (df[u_col] == 0) & (df[ids_col].isna())
            df.loc[condition, ids_col] = "0"

            # Display comparison
            print(f"\n 🔄 Processing: {u_col} ↔ {ids_col}")
            comparison = pd.DataFrame({
                'Before_u': original_u.head(3),
                'After_u': df[u_col].head(3),
                'Before_ids': original_ids.head(3),
                'After_ids': df[ids_col].head(3)
            })
            display(comparison)

            # Rename column
            new_name = f"{ids_col}_cleaned"
            df.rename(columns={ids_col: new_name}, inplace=True)
            renamed_columns[ids_col] = new_name
            print(f" ✨ Renamed to: {new_name}")
            print("─" * 50)

    # Show final columns
    print("\n✅ FINAL COLUMNS PER DATAFRAME:")
    for sheet_name, df in df_dict.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f"\n📌 {sheet_name} columns:")
            cols = [f"{col} {'(renamed)' if col in renamed_columns.values() else ''}"
                   for col in df.columns]
            print("\n".join(f" • {col}" for col in cols))

    return df_dict, renamed_columns

def drop_zero_values_columns(df_dict) -> tuple:
    """
    Verifies missing values, drops columns with 100% missing values,
    and shows only relevant columns for each DataFrame.

    Args:
        df_dict (dict): Dictionary containing DataFrames.

    Returns:
        Tuple: (updated_df_dict, results_dict, dropped_columns_report)
    """
    results_dict = {}
    dropped_columns_report = {}
    updated_df_dict = df_dict.copy()

    # Define the prefix mapping for each DataFrame
    prefix_mapping = {
        "Murs": "Mur_",
        "Sols": "Sol_",
        "Poutres": "Poutre_",
        "Poteaux": "Poteau_"
    }

    for df_name, df in updated_df_dict.items():
        print(f"\n{'='*50}")
        print(f"🔍 ANALYZING: {df_name.upper()}")
        print(f"{'='*50}")

        # Get the prefix for this DataFrame
        prefix = prefix_mapping.get(df_name, "")

        # Find all columns that start with this prefix
        relevant_cols = [col for col in df.columns if col.startswith(prefix)]

        if not relevant_cols:
            print(f"⚠️ No relevant columns found for {df_name}")
            continue

        # Identify columns with 100% missing values
        missing_percent = (df[relevant_cols].isna().mean() * 100)
        cols_to_drop = missing_percent[missing_percent == 100].index.tolist()

        # Drop columns with 100% missing values
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            updated_df_dict[df_name] = df
            dropped_columns_report[df_name] = cols_to_drop
            print(f"\n❌ Dropped columns (100% missing):")
            for col in cols_to_drop:
                print(f" - {col}")
            relevant_cols = [col for col in relevant_cols if col not in cols_to_drop]

        # Create summary for remaining columns
        summary_df = pd.DataFrame({
            "Column": relevant_cols,
            "Missing Count": df[relevant_cols].isna().sum(),
            "Missing %": (df[relevant_cols].isna().mean() * 100).round(2)
        })

        results_dict[df_name] = summary_df

        # Display summary
        print(f"\n📊 Missing Value Summary for {df_name}:")
        display(summary_df.style.format({"Missing %": "{:.2f}%"}).background_gradient(
            subset=["Missing %"], cmap="Reds", vmin=0, vmax=100))

        # Generate missingno plot for relevant columns only
        if relevant_cols:  # Only plot if there are columns left
            plt.figure(figsize=(10, 6))
            msno.bar(df[relevant_cols], figsize=(10, 6), color="dodgerblue", fontsize=12)
            plt.title(f"Missing Data Pattern - {df_name}", pad=20, fontsize=14)
            plt.ylabel("Column", labelpad=15)
            plt.xlabel("Data Completeness", labelpad=15)
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️ No columns remaining after dropping 100% missing columns")

    return updated_df_dict, results_dict, dropped_columns_report

def count_ids_per_row(df_dict) -> tuple:
    """
    Dynamically searches for columns ending with `_coupants_Ids_cleaned` or `_coupés_Ids_cleaned`
    across multiple DataFrames, then counts comma-separated IDs.

    Args:
        df_dict (dict): Dictionary of DataFrames (e.g., {"Murs": murs_df, "Sols": sols_df, ...})

    Returns:
        Updated DataFrames with count columns added and a mapping of processed columns.
    """
    renamed_columns = {}  # Track detected columns

    for df_name, df in df_dict.items():
        print(f"\n🔍 Processing {df_name}...")

        # Find matching columns dynamically
        matching_cols = [col for col in df.columns if col.endswith("_coupants_Ids_cleaned") or col.endswith("_coupés_Ids_cleaned")]

        for new_col in matching_cols:
            renamed_columns[new_col] = new_col  # Store detected column

            # Convert to string and clean empty/nan cases
            df[new_col] = df[new_col].astype(str).replace(['nan', 'None', '0', '', ' '], np.nan)

            # Count comma-separated IDs
            df[f"{new_col}_count"] = df[new_col].apply(lambda x: len(str(x).split(",")) if pd.notna(x) else 0)

            print(f"✅ Found & Processed: {new_col}, Added Count Column: {new_col}_count")

    return df_dict, renamed_columns

def verify_missing_values_with_missingno(df_dict):
    """
    Verifies missing values for relevant columns in each DataFrame using `missingno`.
    Shows only the columns that belong to each specific DataFrame.

    Args:
        df_dict (dict): Dictionary containing DataFrames.

    Returns:
        Dictionary with summary results for each DataFrame.
    """
    results_dict = {}

    # Define the prefix mapping for each DataFrame
    prefix_mapping = {
        "Murs": "Mur_",
        "Sols": "Sol_",
        "Poutres": "Poutre_",
        "Poteaux": "Poteau_"
    }

    for df_name, df in df_dict.items():
        print(f"\n{'='*50}")
        print(f"🔍 ANALYZING: {df_name.upper()}")
        print(f"{'='*50}")

        # Get the prefix for this DataFrame
        prefix = prefix_mapping.get(df_name, "")

        # Find all columns that start with this prefix
        relevant_cols = [col for col in df.columns if col.startswith(prefix)]

        if not relevant_cols:
            print(f"⚠️ No relevant columns found for {df_name}")
            continue

        # Create summary for only relevant columns
        summary_df = pd.DataFrame({
            "Column": relevant_cols,
            "Missing Count": df[relevant_cols].isna().sum(),
            "Missing %": (df[relevant_cols].isna().mean() * 100).round(2)
        })

        results_dict[df_name] = summary_df

        # Display summary
        print(f"\n📊 Missing Value Summary for {df_name}:")
        display(summary_df.style.format({"Missing %": "{:.2f}%"}))

        # Generate missingno plot for relevant columns only
        plt.figure(figsize=(15, 7))
        msno.bar(df[relevant_cols], figsize=(15, 7), color="dodgerblue", fontsize=12)
        plt.title(f"Missing Data Pattern - {df_name}", pad=20, fontsize=14)
        plt.ylabel("Column", labelpad=15)
        plt.xlabel("Data Completeness", labelpad=15)
        plt.tight_layout()
        plt.show()

    return results_dict


### ============================================================================
### Function to sanitize column names by removing spaces and special characters
### ============================================================================
def sanitize_column_name(col_name):
    """Remove spaces and special characters from column names."""
    return col_name.strip().replace(" ", "_").replace("(", "").replace(")", "")


### ============================================================================
### Function to load and choose essential columns from columns in Excel sheets
### ============================================================================
def load_and_sanitize_data(filepath):
    """
    Load, clean, and sanitize all DataFrames from Excel file.
    Returns dictionary of fully sanitized DataFrames.
    """
    dfs = {}

    try:
        print("📂 Loading and sanitizing Excel file...")
        xls = pd.ExcelFile(filepath)

        for sheet, keep_cols in ESSENTIAL_COLUMNS.items():
            if sheet in xls.sheet_names:
                # Load original data
                df = pd.read_excel(filepath, sheet_name=sheet)

                # Clean and sanitize column names
                sanitized_cols = {col: sanitize_column_name(col) for col in df.columns if col in keep_cols}
                prefix = PREFIXES.get(sheet, "")
                renamed_cols = {col: prefix + sanitized_cols[col] for col in sanitized_cols}

                # Create sanitized DataFrame
                dfs[sheet] = df[list(sanitized_cols.keys())].rename(columns=renamed_cols)

                # Also sanitize the data values in ID columns
                for col in dfs[sheet].columns:
                    if col.endswith('_Id') or col.endswith('_Ids'):
                        dfs[sheet][col] = dfs[sheet][col].astype(str).str.strip()

                print(f"✅ {sheet}: Sanitized {len(renamed_cols)} columns")

    except Exception as e:
        print(f"🚨 Error: {str(e)}")
        return {sheet: pd.DataFrame() for sheet in ESSENTIAL_COLUMNS.keys()}

    return dfs
