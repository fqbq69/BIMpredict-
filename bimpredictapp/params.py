import os
import numpy as np

##################  PATH ENV VARIABLES  ##################
EXCEL_FILES_PATH = "raw_data/excel"
A_FILE_TO_TEST = 'raw_data/excel/RawData - 21003_Paramétrage ICF_18-01-22.xlsx'
TF_CPP_MIN_LOG_LEVEL = 3

# Pipeline mode: train, predict or test
MODE = 'test'

# Define project folder paths
# Data directories
BASE_DIR = "bimpredictapp/"
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")
PREDICTED_DATA_DIR = os.path.join(DATA_DIR, "predicting_data")
TESTING_DATA_DIR = os.path.join(DATA_DIR, "testing_data")

# Model directories
MODELS_DIR = os.path.join(BASE_DIR, "models")
ML_MODELS_DIR = os.path.join(MODELS_DIR, "sk/machine_learning")
DL_MODELS_DIR = os.path.join(MODELS_DIR, "sk/deep_learning")
OTHER_MODELS_DIR = os.path.join(MODELS_DIR, "sk/other")

# Python modules and plots directories
PYTHON_MODULES_DIR = os.path.join(BASE_DIR, "python_modules")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

### ===============================================
### Define Models Paths
### ===============================================

ML_MODELS_DIR = os.path.join(MODELS_DIR, "machine_learning")
DL_MODELS_DIR = os.path.join(MODELS_DIR, "deep_learning")

ENCODERS_PATH = os.path.join(MODELS_DIR, "feature_encoders.pkl")
TARGET_ENCODERS_PATH = os.path.join(MODELS_DIR, "target_encoders.pkl")
MODEL_FEATURES_PATH = os.path.join(MODELS_DIR, "model_features.pkl")


### ===============================================
### Define Features
### ===============================================
TARGET_FEATURES = ['Murs', 'Sols', 'Poutres', 'Poteaux']



env_valid_options = dict(
)

def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)
