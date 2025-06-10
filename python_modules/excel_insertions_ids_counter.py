import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def count_ids_per_row(df_dict):
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


def plot_id_counts(df_dict, renamed_columns):
    """
    Generates a bar plot showing the count of IDs per column across all DataFrames.

    Args:
        df_dict (dict): Dictionary of processed DataFrames.
        renamed_columns (dict): Mapping of processed ID columns.
    """
    plot_data = []

    for df_name, df in df_dict.items():
        for col in renamed_columns.values():
            count_col = f"{col}_count"
            if count_col in df.columns:
                total_count = df[count_col].sum()
                plot_data.append((df_name, col, total_count))

    # Convert to DataFrame for easy plotting
    plot_df = pd.DataFrame(plot_data, columns=["DataFrame", "Column", "Total ID Count"])

    # Plot results
    plt.figure(figsize=(12, 6))
    for df_name in plot_df["DataFrame"].unique():
        subset = plot_df[plot_df["DataFrame"] == df_name]
        plt.bar(subset["Column"], subset["Total ID Count"], label=df_name)

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Total ID Count")
    plt.xlabel("Columns")
    plt.title("ID Counts per Column Across DataFrames")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
