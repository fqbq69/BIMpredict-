import pandas as pd

from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

from bmipredictapp.params import *

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """

    print("✅ data cleaned")

    return df
