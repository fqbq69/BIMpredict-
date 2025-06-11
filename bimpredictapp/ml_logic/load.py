
import numpy as np
import pandas as pd
from colorama import Fore, Style

from pathlib import Path
import os
import tensorflow as tf

from params import *

def load_excel() -> dict:
    """
    Load data from an Excel file and return a dictionary of DataFrames.

    Parameters:
    - maquettes_path (str): Path to the Excel file.
    - sheets (list): List of sheet names to load.

    Returns:
    - dict: Dictionary with sheet names as keys and DataFrames as values.
    """

    # List all Excel files in RAW_DATA_DIR
    excel_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".xlsx") or f.endswith(".xls")]

    # Dictionary to store DataFrames for each file and sheet
    dataframes = {}

    # Process each Excel file
    for file in excel_files:
        file_path = os.path.join(RAW_DATA_DIR, file)
        print(f"Loading: {file_path}")

        try:
            # Load Excel file
            excel_data = pd.ExcelFile(file_path)

            # Load all sheets dynamically
            for sheet_name in excel_data.sheet_names:
                df = excel_data.parse(sheet_name)

                # Save DataFrame with a unique identifier
                dataframes[f"{file}_{sheet_name}"] = df

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Display summary of loaded data
    print(f"\nTotal files processed: {len(dataframes)}")

    for key, df in dataframes.items():
        print(f"Loaded DataFrame: {key}, Shape: {df.shape}")

    return dataframes
