import pandas as pd
import numpy as np

#this module verifies the names of the sheets.
from bimpredictapp.ml_logic.load import load_excel_files


def get_sheets(file_url:str)-> dict:

    try:
        file = pd.ExcelFile(file_url)
        return file.sheet_names

    except Exception as e:
        print(f"Error loading sheets from file: {e}")
