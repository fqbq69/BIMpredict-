import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from IPython.display import display
try:
    from excel_merge_dfs import merged_dfs_dict
except ImportError:
    print("❌ Error: Unable to import 'excel_merge_dfs'. Ensure the module is in the correct directory.")
    merged_dfs_dict = {}

def verify_missing_values_merged(maquette_name):
    """
    Verifies missing values in the dynamically named merged DataFrame.
    """
    merged_variable_name = f"{maquette_name}_merged_v1"

    # Debugging step: Print available keys
    print(f"🔍 Looking for '{merged_variable_name}' in merged_dfs_dict")
    print(f"Available merged DataFrames: {merged_dfs_dict.keys()}")

    if merged_variable_name not in merged_dfs_dict:
        print(f"❌ Error: Merged DataFrame '{merged_variable_name}' not found in merged_dfs_dict.")
        return None

    merged_df = merged_dfs_dict[merged_variable_name]

    return merged_df

def missing_values_table(maquette_name):
    """
    Generates a structured table showing column names, total rows, missing values, and percentage of missing values.
    """
    merged_variable_name = f"{maquette_name}_merged_v1"

    if merged_variable_name not in merged_dfs_dict:
        print(f"❌ Error: Merged DataFrame '{merged_variable_name}' not found in merged_dfs_dict.")
        return None

    merged_df = merged_dfs_dict[merged_variable_name]

    # Create structured summary table
    missing_summary = pd.DataFrame({
        "Column Name": merged_df.columns,
        "Total Rows": [merged_df.shape[0]] * len(merged_df.columns),
        "Missing Values": merged_df.isna().sum().values,
        "Percentage Missing": (merged_df.isna().mean() * 100).round(2).values
    })

    # Display the formatted table
    display(missing_summary)

    return missing_summary


def plot_missing_values(maquette_name):
    """
    Plots the missing values in the merged DataFrame.
    """
    merged_variable_name = f"{maquette_name}_merged_v1"

    if merged_variable_name not in merged_dfs_dict:
        print(f"❌ Error: Merged DataFrame '{merged_variable_name}' not found in merged_dfs_dict.")
        return None

    merged_df = merged_dfs_dict[merged_variable_name]

    # Generate missingno plot
    plt.figure(figsize=(12, 6))
    msno.bar(merged_df, figsize=(12, 6), color="dodgerblue", fontsize=12)
    plt.title(f"Missing Data Pattern - {merged_variable_name}", pad=20, fontsize=14)
    plt.ylabel("Column", labelpad=15)
    plt.xlabel("Data Completeness", labelpad=15)
    plt.tight_layout()
    plt.show()
