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


### ===============================================
### PREPARING DIRECTORIES
### ===============================================
directories = [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, PREDICTED_DATA_DIR,
    MODELS_DIR, ML_MODELS_DIR, DL_MODELS_DIR, OTHER_MODELS_DIR,
    PYTHON_MODULES_DIR, PLOTS_DIR
]

# Checking directories and creating directories if they don't exist
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")


### ===============================================
### MODE SELECTION
### ===============================================

    if MODE == 'train':
        excel_files_list = RAW_DATA_DIR
        print(RAW_DATA_DIR)
    elif MODE == 'predict':
        excel_files_list = PREDICTED_DATA_DIR
    elif MODE == 'test':
        excel_files_list = TESTING_DATA_DIR
        print(TESTING_DATA_DIR)

### ===============================================
### Loading all excel file(s) in a directory
### ===============================================
def list_excel_files(files_path)-> list:
    '''
    Call this function to get all filenames in a folder into one list, for training purpose.
    '''
    #old: all_files = [f for f in listdir(files_path) if isfile(join(files_path, f))]

    excel_files = [f for f in os.listdir(files_path) if f.endswith(".xlsx") or f.endswith(".xls")]
    return excel_files

### ===============================================
### Convert Excel file(s) to dataframes
### ===============================================

def load_excel_files(excel_files) -> None:
    """
    Loading excel files from the directory defined in the env variables as a dataframe

    """
    dataframes = load_excel(excel_files)

    print("✅ Loading the maquettes into dataframes is done \n")

    return dataframes

### ===============================================
### Preprocess the data
### ===============================================

def preprocess() -> None:
    """
    Prepares the data from each sheet and calls the required functions
    before training or using the models in the predicting process.
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    print("✅ preprocess() done \n")
    pass

### ===============================================
### Train the model(s)
### ===============================================

def train(
        split_ratio: float = 0.02, # 0.02 represents ~ 1 month of validation data on a 2009-2015 train set
        learning_rate=0.0005,
        batch_size = 256,
        patience = 2
    ) -> float:

    """
    Splitting the data into train/val/test
    Trainging the models if rewuired, with the sheets.
    Saves the model after training
    moves the last model into stagin
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    # Load processed data using `get_data_with_cache` in chronological order
    # Try it out manually on console.cloud.google.com first!

    # Create (X_train_processed, y_train, X_val_processed, y_val)

    # Train model using `model.py`

    # Save results on the hard drive using taxifare.ml_logic.registry

    # Save model weight on the hard drive (and optionally on GCS too!)

    # The latest model should be moved to staging

    print("✅ train() done \n")

    pass #return the score here

### ===============================================
### Evaluate Models
### ===============================================

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

### ===============================================
### Load and predict
### ===============================================

def pred(file: str) -> str:
    """
    Make a prediction using the latest trained model
    """

    #print("\n⭐️ Use case: predict")


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
    excel_file = list_excel_files(excel_files_list)
    load_excel_files(excel_file)
    #preprocess()
    #train()
    #evaluate()
    #pred()
