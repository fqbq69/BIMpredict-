
import numpy as np
import pandas as pd
from colorama import Fore, Style
import os

from bimpredictapp.params import *

def load_excel_file(excel_files_path, sheets) -> dict: #dict of four features
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
            dfs[sheet] = pd.read_excel(excel_files_path, sheet_name=sheet)
            print(f"{sheet} loaded successfully.")

        except Exception as e:
            print(f"Error loading data: {e}")
            # Handle missing sheets
            available_sheets = pd.ExcelFile(excel_files_path).sheet_names
            print(f"Available sheets in this file: {available_sheets}")
            for sheet in sheets:
                if sheet in available_sheets:
                    dfs[sheet] = pd.read_excel(excel_files_path, sheet_name=sheet)
                else:
                    print(f"Sheet '{sheet}' not found in the Excel file.")
                    dfs[sheet] = pd.DataFrame()  # Empty DataFrame if sheet is missing

    return dfs
