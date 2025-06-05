import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from IPython.display import display

def drop_zero_values_columns(df_dict):
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
