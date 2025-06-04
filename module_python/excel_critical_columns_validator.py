import pandas as pd
### ====================
### Define Critical Columns
### ====================
CRITICAL_COLUMNS = ["011EC_Lot", "012EC_Ouvrage", "013EC_Localisation", "014EC_Mode Constructif"]

### ====================
### Verify Critical Columns
### ====================
def validate_critical_columns(dataframes):
    """Verify that all critical columns are present in each DataFrame."""
    for sheet, df in dataframes.items():
        missing_critical = [col for col in CRITICAL_COLUMNS if col not in df.columns]
        if missing_critical:
            print(f"🚨 Critical columns missing in {sheet}: {missing_critical}")

    murs_df = dataframes.get('Murs', pd.DataFrame())
    sols_df = dataframes.get('Sols', pd.DataFrame())
    poutres_df = dataframes.get('Poutres', pd.DataFrame())
    poteaux_df = dataframes.get('Poteaux', pd.DataFrame())

    return murs_df, sols_df, poutres_df, poteaux_df
