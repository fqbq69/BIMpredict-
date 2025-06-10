import pandas as pd
import os
def find_columns_with_keyword(df, keyword="_ids_cleaned_count"):
    """
    Finds and displays columns that contain a specific keyword in their names.

    Args:
        df (pd.DataFrame): The DataFrame to scan.
        keyword (str): The keyword to search for in column names.

    Returns:
        List of matching column names.
    """
    matching_columns = [col for col in df.columns if keyword in col]

    if matching_columns:
        print(f"\n🔍 Columns containing '{keyword}':")
        for col in matching_columns:
            print(f" - {col}")
    else:
        print(f"\n✅ No columns found with '{keyword}' in their names.")

    return matching_columns

def drop_columns_with_keyword(df, keyword="_ids_cleaned_count"):
    """
    Identifies and drops columns containing a specific keyword in their names.
    Displays the DataFrame shape before and after the removal.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        keyword (str): The keyword to search for in column names.

    Returns:
        Cleaned DataFrame.
    """
    # Find columns matching the keyword
    matching_columns = [col for col in df.columns if keyword in col]

    # Display original shape
    original_shape = df.shape
    print(f"\n📏 Original shape: {original_shape}")

    if matching_columns:
        print(f"\n❌ Dropping columns containing '{keyword}':")
        for col in matching_columns:
            print(f" - {col}")

        # Drop columns
        cleaned_df = df.drop(columns=matching_columns)

        # Display new shape after removal
        new_shape = cleaned_df.shape
        print(f"\n📏 New shape after removal: {new_shape}")

        return cleaned_df
    else:
        print(f"\n✅ No columns found with '{keyword}'. No changes made.")
        return df

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
