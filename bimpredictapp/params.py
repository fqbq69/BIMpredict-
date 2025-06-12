import os
import numpy as np

##################  TESTING ONLY PATH ENV VARIABLES  ##################
EXCEL_FILES_PATH = "../../raw_data/excel"
A_FILE_TO_TEST = '../../raw_data/excel/RawData - 21003_Paramétrage ICF_18-01-22.xlsx'
TF_CPP_MIN_LOG_LEVEL = 3

MODEL_TEST_DIR = "bimpredictapp/models/machine_learning/pipeline_randomforestpoteauxpipeline.pkl"

#app mode : 'training' or 'predicting'
MODE = 'predicting'


# Data directories
BASE_DIR = "bimpredictapp/"
DATA_DIR = os.path.join(BASE_DIR, "data")

#USE ONE OF THESE TO TEST EXCEL FILES
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

# X FEATURES
X_FEATURES = [
    "011EC_Lot",
    "012EC_Ouvrage",
    "013EC_Localisation",
    "014EC_Mode Constructif",
    "Epaisseur",
    "Sols en intersection",
    "Sols coupés (u)",
    "Sols coupants (u)",
    "Sol au-dessus",
    "Sol en-dessous",
    "Fenêtres",
    "Portes",
    "Ouvertures",
    "Murs imbriqués",
    "Mur multicouche",
    "Profil modifié",
    "Extension inférieure",
    "Extension supérieure",
    "Partie inférieure attachée",
    "Partie supérieure attachée",
    "Décalage supérieur",
    "Décalage inférieur",
    "Matériau structurel",
    "Famille et type",
    "Nom",
]

TARGET_FEATURES= [
    "011ec_lot",
    "012ec_ouvrage",
    "013ec_localisation",
    "014ec_mode_constructif"
]
EXCEL_SHEETS = ['Murs', 'Sols', 'Poutres', 'Poteaux']

env_valid_options = dict(
)

def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)
