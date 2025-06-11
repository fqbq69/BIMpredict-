### ====================
### Importing libraries
### ====================
# %matplotlib inline
import openpyxl
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import joblib
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

#fix missing modules to import
from bimpredictapp.ml_logic.encoders import *


def preprocess_features(X: pd.DataFrame) -> np.ndarray:

    base_path = '../data/raw/'
    processed_path = '../data/processed/'
    os.makedirs(processed_path, exist_ok=True)
    ### ====================
    ### Load and merge data from multiple sheets in an Excel file
    ### ====================

    file_path = "../data/raw/maquette_23001.xlsx"
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}\nPlease check that the file exists at the specified path. Skipping data loading and merging.")
    else:
        dfs = pd.read_excel(file_path, sheet_name=None)

        # Extract individual sheets with validation
        required_sheets = ["Murs", "Sols", "Poteaux", "Poutres"]
        for sheet in required_sheets:
            if sheet not in dfs:
                raise ValueError(f"Required sheet '{sheet}' is missing in the Excel file")

        mur_df = dfs["Murs"].copy()
        sol_df = dfs["Sols"].copy()
        poteau_df = dfs["Poteaux"].copy()
        poutre_df = dfs["Poutres"].copy()

        # Step 1: Clean column names (strip whitespace)
        for df in [mur_df, sol_df, poteau_df, poutre_df]:
            df.columns = df.columns.str.strip()

        # Step 2: Prepare ID columns for merging
        def safe_split(x):
            return str(x).split(",") if pd.notna(x) else []

        # Process mur_df -> sol relationships
        mur_df["Sols_coupants_Ids"] = mur_df["Sols coupants (Ids)"].apply(safe_split)
        mur_df = mur_df.explode("Sols_coupants_Ids").rename(columns={"Sols_coupants_Ids": "Sol_ID"})

        # Ensure both Sol_ID and Id are string type for merging
        mur_df["Sol_ID"] = mur_df["Sol_ID"].astype(str)
        sol_df["Id"] = sol_df["Id"].astype(str)

        # Process sol_df -> poteau/poutre relationships
        sol_df["Poteaux_IDs"] = sol_df["Poteaux coupés (Ids)"].apply(safe_split)
        sol_df["Poutres_IDs"] = sol_df["Poutres coupés (Ids)"].apply(safe_split)

        # Step 3: First merge (mur -> sol)
        merged_df = mur_df.merge(
            sol_df,
            left_on="Sol_ID",
            right_on="Id",
            how="left",
            suffixes=("_mur", "_sol")
        )

        # Step 4: Explode the poteau and poutre relationships
        merged_df = merged_df.explode("Poteaux_IDs").explode("Poutres_IDs")

        # Step 5: Prepare for second merge (sol -> poteau)
        poteau_df = poteau_df.rename(columns={"Id": "Poteau_Id"})
        # Ensure both columns are string type for merging
        merged_df["Poteaux_IDs"] = merged_df["Poteaux_IDs"].astype(str)
        poteau_df["Poteau_Id"] = poteau_df["Poteau_Id"].astype(str)
        merged_df = merged_df.merge(
            poteau_df,
            left_on="Poteaux_IDs",
            right_on="Poteau_Id",
            how="left",
            suffixes=("", "_poteau")
        )

        # Step 6: Prepare for third merge (sol -> poutre)
        poutre_df = poutre_df.rename(columns={"Id": "Poutre_Id"})  # This was the missing step

        # Ensure both columns are string type for merging
        merged_df["Poutres_IDs"] = merged_df["Poutres_IDs"].astype(str)
        poutre_df["Poutre_Id"] = poutre_df["Poutre_Id"].astype(str)

        merged_df = merged_df.merge(
            poutre_df,
            left_on="Poutres_IDs",
            right_on="Poutre_Id",
            how="left",
            suffixes=("", "_poutre")
        )

        # Step 7: Clean up columns
        # # Keep only relevant columns or rename duplicates
        # final_columns = [
        #     'Id_mur', 'Nom_mur', 'Sol_ID', 'Id_sol', 'Nom_sol',
        #     'Poteaux_IDs', 'Poteau_Id', 'Nom_poteau',
        #     'Poutres_IDs', 'Poutre_Id', 'Nom_poutre'
        # ]

        # # Select only existing columns
        # final_columns = [col for col in final_columns if col in merged_df.columns]
        # merged_df = merged_df[final_columns]

        print("Successfully merged dataset:")
        print(merged_df.head())
        print(f"\nFinal shape: {merged_df.shape}")



        print("✅ X_processed, with shape", X_processed.shape)

    pass #return X_processed data
