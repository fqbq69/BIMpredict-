import pandas as pd

def load_data(maquettes_path, sheets=['Murs', 'Sols', 'Poutres', 'Poteaux']):
    """
    Load data from an Excel file and return a dictionary of DataFrames.

    Parameters:
    - maquettes_path (str): Path to the Excel file.
    - sheets (list): List of sheet names to load.

    Returns:
    - dict: Dictionary with sheet names as keys and DataFrames as values.
    """
    dfs = {}

    try:
        for sheet in sheets:
            dfs[sheet] = pd.read_excel(maquettes_path, sheet_name=sheet)
        print("Data loaded successfully from the Excel file.")
    except Exception as e:
        print(f"Error loading data: {e}")
        # Handle missing sheets
        available_sheets = pd.ExcelFile(maquettes_path).sheet_names
        print(f"Available sheets: {available_sheets}")
        for sheet in sheets:
            if sheet in available_sheets:
                dfs[sheet] = pd.read_excel(maquettes_path, sheet_name=sheet)
            else:
                print(f"Sheet '{sheet}' not found in the Excel file.")
                dfs[sheet] = pd.DataFrame()  # Empty DataFrame if sheet is missing

    return dfs
