import numpy as np
import pandas as pd
from pathlib import Path

import os
import tensorflow as tf


def preprocess() -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Retrieve data `

    # Process data

    # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]


    print("✅ preprocess() done \n")

@mlflow_run
def train(
        split_ratio: float = 0.02, # 0.02 represents ~ 1 month of validation data on a 2009-2015 train set
        learning_rate=0.0005,
        batch_size = 256,
        patience = 2
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    # Load processed data using `get_data_with_cache` in chronological order
    # Try it out manually on console.cloud.google.com first!

    # Below, our columns are called ['_0', '_1'....'_66'] on BQ, student's column names may differ

    # Create (X_train_processed, y_train, X_val_processed, y_val)

    # Train model using `model.py`

    # Save results on the hard drive using taxifare.ml_logic.registry

    # Save model weight on the hard drive (and optionally on GCS too!)

    # The latest model should be moved to staging

    print("✅ train() done \n")

    pass #return the score here


@mlflow_run
def evaluate(stage: str = "Production"
    ) -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return MAE as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    #code in here

    print("✅ evaluate() done \n")

    pass #returning eval values


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    #code here

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    pass #treturn the predicted values


if __name__ == '__main__':
    #preprocess()
    #train()
    #evaluate()
    #pred()
    pass
