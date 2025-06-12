import math
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import joblib

from bimpredictapp.params import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def encode_features(final_cleaned_dataframes) -> pd.DataFrame:
    # Initialize dictionaries to store encoders
    feature_encoders, target_encoders = {}, {}

    print("🚀 Applying categorical encoding across all datasets...")

    for df_name, df in final_cleaned_dataframes.items():
        print(f"\n🔄 Processing {df_name}...")

        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns
        target_cols = [col for col in categorical_cols if col in TARGET_COLUMNS]
        feature_cols = list(set(categorical_cols) - set(target_cols))  # Exclude target columns

        # Encode target columns
        for col in target_cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            target_encoders[f"{df_name}_{col}"] = encoder
            print(f"✅ Target Encoder stored for {df_name} - {col}")

        # Encode feature columns using Label Encoding
        one_hot_cols = []
        for col in feature_cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            feature_encoders[f"{df_name}_{col}"] = encoder
            print(f"✅ Feature Encoder stored for {df_name} - {col}")
            one_hot_cols.append(col)

        # Apply One-Hot Encoding to relevant categorical features
        if one_hot_cols:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded_df = pd.DataFrame(
                encoder.fit_transform(df[one_hot_cols]),
                index=df.index,
                columns=encoder.get_feature_names_out(one_hot_cols)
            )
            df.drop(columns=one_hot_cols, inplace=True)
            df = pd.concat([df, encoded_df], axis=1)

        # Save the updated dataframe
        final_cleaned_dataframes[df_name] = df
        print(f"✅ Completed categorical encoding for {df_name}. Updated shape: {df.shape}")

    print("🎯 Final categorical encoding applied successfully across all datasets!")

    return final_cleaned_dataframes



ML_MODELS_DIR = os.path.join(MODELS_DIR, "machine_learning")
DL_MODELS_DIR = os.path.join(MODELS_DIR, "deep_learning")

ENCODERS_PATH = os.path.join(MODELS_DIR, "feature_encoders.pkl")
TARGET_ENCODERS_PATH = os.path.join(MODELS_DIR, "target_encoders.pkl")
MODEL_FEATURES_PATH = os.path.join(MODELS_DIR, "model_features.pkl")

def save_encoders(feature_encoders, target_encoders):
    """Saves feature and target encoders for consistent data preprocessing."""
    joblib.dump(feature_encoders, ENCODERS_PATH)
    joblib.dump(target_encoders, TARGET_ENCODERS_PATH)

def load_encoders():
    """Loads stored feature and target encoders."""
    return (
        joblib.load(ENCODERS_PATH) if os.path.exists(ENCODERS_PATH) else {},
        joblib.load(TARGET_ENCODERS_PATH) if os.path.exists(TARGET_ENCODERS_PATH) else {}
    )

def encode_new_data(X_new, feature_encoders):
    """Encodes categorical features using stored encoders."""
    for col, encoder in feature_encoders.items():
        if col in X_new:
            X_new[col] = encoder.transform(X_new[col].astype(str))
    return X_new
