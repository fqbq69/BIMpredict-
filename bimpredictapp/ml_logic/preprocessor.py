import numpy as np
import pandas as pd

from colorama import Fore, Style

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

#fix missing modules to import
from bmipredictapp.ml_logic.encoders import transform_time_features, transform_lonlat_features, compute_geohash


def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    def create_sklearn_preprocessor() -> ColumnTransformer:
        """
        Scikit-learn pipeline that transforms a cleaned dataset of shape (_, 7)
        into a preprocessed one of fixed shape (_, 65).

        Stateless operation: "fit_transform()" equals "transform()".
        """

    print("✅ X_processed, with shape", X_processed.shape)

    pass #return X_processed data
