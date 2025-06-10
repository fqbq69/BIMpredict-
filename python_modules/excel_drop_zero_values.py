import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from IPython.display import display
try:
    from excel_merge_dfs import merged_dfs_dict
except ImportError:
    print("❌ Error: Unable to import 'excel_merge_dfs'. Ensure the module is in the correct directory.")
    merged_dfs_dict = {}

def drop_zero_values_columns_merged(maquette_name):
    """
    Verifies missing values, drops columns with 100% missing values,
    and displays the shape of the merged DataFrame before and after cleanup.

    Args:
        maquette_name (str): The name of the maquette to retrieve the correct merged DataFrame.

    Returns:
        Tuple: (cleaned_df, summary_df, dropped_columns)
    """
    merged_variable_name = f"{maquette_name}_merged_v1"

    if merged_variable_name not in merged_dfs_dict:
        print(f"❌ Error: Merged DataFrame '{merged_variable_name}' not found in merged_dfs_dict.")
        return None, None, None

    merged_df = merged_dfs_dict[merged_variable_name]  # ✅ Retrieve the correct DataFrame

    print(f"\n{'='*50}")
    print(f"🔍 ANALYZING: {merged_variable_name}")
    print(f"{'='*50}")

    # Display original shape
    original_shape = merged_df.shape
    print(f"📏 Original shape of {merged_variable_name}: {original_shape}")

    # Identify columns with 100% missing values
    missing_percent = (merged_df.isna().mean() * 100)
    cols_to_drop = missing_percent[missing_percent == 100].index.tolist()

    # Drop columns with 100% missing values
    cleaned_df = merged_df.drop(columns=cols_to_drop)

    # Display new shape after column removal
    new_shape = cleaned_df.shape
    print(f"📏 New shape of {merged_variable_name} after cleanup: {new_shape}")

    # Create summary for remaining columns
    summary_df = pd.DataFrame({
        "Column": cleaned_df.columns,
        "Missing Count": cleaned_df.isna().sum(),
        "Missing %": (cleaned_df.isna().mean() * 100).round(2)
    })

    # Display summary
    print(f"\n📊 Missing Value Summary for {merged_variable_name}:")
    display(summary_df.style.format({"Missing %": "{:.2f}%"}).background_gradient(
        subset=["Missing %"], cmap="Reds", vmin=0, vmax=100))

    # Generate missingno plot for the remaining columns
    plt.figure(figsize=(12, 6))
    msno.bar(cleaned_df, figsize=(12, 6), color="dodgerblue", fontsize=12)
    plt.title(f"Missing Data Pattern - {merged_variable_name}", pad=20, fontsize=14)
    plt.ylabel("Column", labelpad=15)
    plt.xlabel("Data Completeness", labelpad=15)
    plt.tight_layout()
    plt.show()

    return cleaned_df, summary_df, cols_to_drop

import os

def export_cleaned_dataframe(cleaned_df, maquette_name, save_path):
    """
    Exports the cleaned DataFrame as '{maquette_name}_merged_v2.xlsx'.

    Args:
        cleaned_df (pd.DataFrame): The cleaned DataFrame to export.
        maquette_name (str): The name of the maquette.
        save_path (str): Directory to save the cleaned DataFrame.

    Returns:
        save_file_path (str): Path where the cleaned DataFrame was saved.
    """
    if cleaned_df is None:
        print(f"❌ Error: No cleaned DataFrame available for '{maquette_name}'.")
        return None

    cleaned_variable_name = f"{maquette_name}_merged_v2"
    save_file_path = os.path.join(save_path, f"{cleaned_variable_name}.xlsx")

    cleaned_df.to_excel(save_file_path, index=False)
    print(f"✅ Cleaned DataFrame '{cleaned_variable_name}' saved to: {save_file_path}")

    return save_file_path
