import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from bimpredictapp.params import *


# Function to remove low-variance & highly correlated features
def optimize_feature_selection(df, variance_threshold=0.02, correlation_threshold=0.98):
    print(f"\n🔍 Processing {df.shape[0]} rows & {df.shape[1]} columns")

    # Step 1: Remove Low-Variance Features
    selector = VarianceThreshold(variance_threshold)
    numeric_df = df.select_dtypes(include=["number"])  # Focus only on numerical columns
    selector.fit(numeric_df)

    low_variance_cols = numeric_df.columns[~selector.get_support()]
    keep_cols = [col for col in low_variance_cols if any(keyword in col.lower() for keyword in ["coupés", "coupants"])]
    drop_cols = [col for col in low_variance_cols if col not in keep_cols and col not in TARGET_COLUMNS]

    df.drop(columns=drop_cols, inplace=True)
    print(f"⚠️ Dropped {len(drop_cols)} low-variance columns (excluding 'coupés' and target columns): {drop_cols}")

    # Step 2: Remove Highly Correlated Features
    numeric_df = df.select_dtypes(include=["number"])
    correlation_matrix = numeric_df.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    correlated_features = [
        col for col in upper_triangle.columns
        if any(upper_triangle[col] > correlation_threshold) and col not in TARGET_COLUMNS
    ]

    df.drop(columns=correlated_features, inplace=True)
    print(f"⚠️ Dropped {len(correlated_features)} highly correlated columns (excluding target columns): {correlated_features}")

    print(f"✅ Final shape after filtering: {df.shape}")
    return df

# Apply optimized feature selection to all datasets

def optimize_feature_selection_apply(final_cleaned_dataframes):
    optimized_cleaned_dataframes = {name: optimize_feature_selection(df) for name, df in final_cleaned_dataframes.items()}
    print("🚀 Optimized feature selection completed successfully!")
    return optimized_cleaned_dataframes
