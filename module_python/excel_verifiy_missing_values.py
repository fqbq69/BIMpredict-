import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from IPython.display import display

def verify_missing_values_with_missingno(df_dict) -> dict:
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
