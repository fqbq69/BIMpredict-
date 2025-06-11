import pandas as pd
import numpy as np
from IPython.display import display, HTML

def clean_ids_columns(df_dict):
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
