
import numpy as np
import pandas as pd
from colorama import Fore, Style

from pathlib import Path
import os
import tensorflow as tf


# Display column names
for sheet_name, df in data.items():
    print(f"\n{sheet_name} DataFrame Preview:")
    print(df.columns)


def load_excel_files(excel_file_path, sheets=['Murs', 'Sols', 'Poutres', 'Poteaux']) -> dict:
    """
    Load data from an Excel file and return a dictionary of DataFrames.

    Parameters:
    - maquettes_path (str): Path to the Excel file.
    - sheets (list): List of sheet names to load.

    Returns:
    - dict: Dictionary with sheet names as keys and DataFrames as values.
    """
    dfs = {}

    for sheet in sheets:
        try:
            dfs[sheet] = pd.read_excel(excel_file_path, sheet_name=sheet)
            print(f"{sheet} loaded successfully.")

        except Exception as e:
            print(f"Error loading data: {e}")
            # Handle missing sheets
            available_sheets = pd.ExcelFile(excel_file_path).sheet_names
            print(f"Available sheets in this file: {available_sheets}")
            for sheet in sheets:
                if sheet in available_sheets:
                    dfs[sheet] = pd.read_excel(excel_file_path, sheet_name=sheet)
                else:
                    print(f"Sheet '{sheet}' not found in the Excel file.")
                    dfs[sheet] = pd.DataFrame()  # Empty DataFrame if sheet is missing

    return dfs
