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


# Function to convert ID strings into a numeric count feature
def count_ids(id_string):
    """Convert string of IDs into a numeric count."""
    return len(id_string.split(",")) if isinstance(id_string, str) else 0




# Apply processing to fully cleaned datasets
###################################################
# Preproccsing Data
###################################################

def preprocess(final_cleaned_dataframes) -> pd.DataFrame:
    for df_name, df in final_cleaned_dataframes.items():
        print(f"\n🔄 Processing ID count transformation for {df_name}...")

        # Identify relevant ID columns
        id_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ["coupés_(ids)", "coupants_(ids)"])]

        if id_columns:
            print(f"📌 Found ID columns: {id_columns}")

            # Transform ID columns into numeric count and drop originals
            df[[f"{col}_count" for col in id_columns]] = df[id_columns].applymap(count_ids)
            df.drop(columns=id_columns, inplace=True)  # Remove original text-based ID columns

        # Ensure only ID-related columns are converted to numeric
        df[id_columns] = df[id_columns].apply(pd.to_numeric, errors="coerce").fillna(0)

        # Store the updated dataframe
        final_cleaned_dataframes[df_name] = df

        print(f"✅ Final shape after ID count transformation: {df.shape}")

    print("🚀 ID count transformation completed successfully!")

    return final_cleaned_dataframes
