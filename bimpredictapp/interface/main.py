import numpy as np
import pandas as pd
from colorama import Fore, Style
import os
from pathlib import Path

from bimpredictapp.params import *
from bimpredictapp.ml_logic.load import load_excel
from bimpredictapp.ml_logic.data import clean_ids_columns, drop_zero_values_columns
from bimpredictapp.ml_logic.data import count_ids_per_row, verify_missing_values_with_missingno

import tensorflow as tf
from tensorflow import keras

from os import listdir
from os.path import isfile, join

def load_all_files():
    all_files = [f for f in listdir(EXCEL_FILES_PATH) if isfile(join(EXCEL_FILES_PATH, f))]
    return all_files

def import_excel_files() -> None:
    """
    Loading excel files from the directory defined in the env variables

    """
    df_dict = load_excel(maquettes_path = A_FILE_TO_TEST, sheets=['Murs', 'Sols', 'Poutres', 'Poteaux'])

    print("✅ Loading the maquette done \n")

    pass

def preprocess() -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    print("✅ preprocess() done \n")
    pass

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

def pred(file: str) -> str:
    """
    Make a prediction using the latest trained model
    """

    #print("\n⭐️ Use case: predict")

    #load new sheets as dataframes
    df_dict = load_excel(maquettes_path = A_FILE_TO_TEST, sheets=['Murs', 'Sols', 'Poutres', 'Poteaux'])

    #Preprocess the new loaded df
    #clean_df_dict = proc(df_dict)


    #load model(s) to make prediction - using a for loop tp 4 models prediction process

    #target_model = tf.keras.models.load_model('/models/target_model.keras')

    #predict results
    '''
    murs_pred = target_model1.predict(clean_df_dict['Murs'])
    sols_pred = target_model2.predict(clean_df_dict['Sols'])
    poutres_pred = target_model2.predict(clean_df_dict['Poutres'])
    poteaux_pred = target_model2.predict(clean_df_dict['Poteaux'])

    #merge results to main db_dict
    murs_full= pd.concat(df_dict['Murs'], murs_pred)
    sols_full= pd.concat(df_dict['Sols'], sols_pred)
    poutres_full= pd.concat(df_dict['Poutres'], poutres_pred)
    poteaux_full= pd.concat(df_dict['Poteaux'], poteaux_pred)

    #remake an excel file from dfs

    with pd.ExcelWriter('/data/output/predicted.xlsx') as writer:
        murs_full.to_excel(writer, sheet_name='Murs')
        sols_full.to_excel(writer, sheet_name='Sols')
        sols_full.to_excel(writer, sheet_name='Poutres')
        sols_full.to_excel(writer, sheet_name='Poteaux')

    #return the url of the file to download

    #print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    '''
    return  'prediction completed, you can download the predicted.xlsx'

if __name__ == '__main__':
    #load_all_files()
    #import_excel_files()
    #preprocess()
    #train()
    #evaluate()
    #pred()
    pass
