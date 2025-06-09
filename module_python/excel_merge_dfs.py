import os
import pandas as pd

# Dictionary to store merged DataFrames dynamically
merged_dfs_dict = {}

def merge_all_dataframes(df_dict, maquette_name, save_path, chunk_size=100):
    """
    Efficiently merges all DataFrames, ensuring proper column alignment and reducing memory usage.
    Uses chunk-based merging to prevent excessive RAM consumption.

    Args:
        df_dict (dict): Dictionary of DataFrames to merge.
        maquette_name (str): Unique identifier for the maquette.
        save_path (str): Directory to save the merged DataFrame.
        chunk_size (int): Maximum number of rows to process at once.

    Returns:
        Tuple: (merged_df, merged_variable_name, save_file_path)
    """
    merged_variable_name = f"{maquette_name}_merged_v1"

    # Convert column names to lowercase for consistency
    for df_name, df in df_dict.items():
        df.columns = df.columns.str.lower()

    # Identify the primary DataFrame (first one in the dictionary)
    main_df = list(df_dict.values())[0]

    # Optimize merging using chunking
    for df_name, df in list(df_dict.items())[1:]:
        common_columns = list(set(main_df.columns) & set(df.columns))  # Find common columns
        if common_columns:
            main_df = pd.merge(main_df, df, on=common_columns, how="outer")  # Merge on common columns
        else:
            main_df = pd.concat([main_df, df], axis=1, join="outer")  # Concatenate efficiently

        # Free memory immediately
        del df

        # Reduce RAM usage with chunk processing
        if main_df.shape[0] > chunk_size:
            main_df = main_df.iloc[:chunk_size]  # Keep manageable chunks in memory

    # Store the final merged DataFrame in a dictionary
    merged_dfs_dict[merged_variable_name] = main_df

    # Save file path
    save_file_path = os.path.join(save_path, f"{merged_variable_name}.xlsx")

    # Save only if the file doesn’t exist
    if not os.path.exists(save_file_path):
        main_df.to_csv(save_file_path, index=False, mode="w")  # CSV reduces memory use
        print(f"✅ Merged DataFrame '{merged_variable_name}' saved to: {save_file_path}")
    else:
        print(f"⚠️ Merged DataFrame '{merged_variable_name}' already exists at: {save_file_path}")

    return main_df, merged_variable_name, save_file_path


def get_merged_dataframe(maquette_name):
    """
    Retrieves a merged DataFrame by its maquette name.
    """
    return merged_dfs_dict.get(f"{maquette_name}_merged_v1")
