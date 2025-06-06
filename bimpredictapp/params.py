import os
import numpy as np

##################  VARIABLES  ##################
DATA_PATH = "raw_data/excel"
MAQ_TO_TEST = 'raw_data/excel/RawData - 21003_Paramétrage ICF_18-01-22.xlsx'



### ===============================================
### Define essential columns for each DataFrame
### ===============================================

ESSENTIAL_COLUMNS = {
    "Murs": [
        "Id", "011EC_Lot", "012EC_Ouvrage", "013EC_Localisation", "014EC_Mode Constructif", "Hauteur", "Epaisseur", "AI", "AS", "Sols en intersection", "Sols coupés (u)", "Sols coupés (Ids)",
        "Sols coupants (u)", "Sols coupants (Ids)", "Sol au-dessus", "Sol en-dessous", "Fenêtres", "Portes", "Ouvertures", "Murs imbriqués",
        "Mur multicouche", "Profil modifié", "Extension inférieure", "Extension supérieure", "Volume", "Surface", "Partie inférieure attachée", "Partie supérieure attachée",
        "Décalage supérieur", "Décalage inférieur", "Matériau structurel",
    ],
    "Sols": [
        "Id", "011EC_Lot", "012EC_Ouvrage", "013EC_Localisation", "014EC_Mode Constructif", "Murs en intersection",
        "Murs coupés (u)", "Murs coupés (Ids)", "Murs coupants (u)", "Murs coupants (Ids)", "Poutres en intersection", "Poutres coupés (u)",
        "Poutres coupés (Ids)", "Poutres coupants (u)", "Poutres coupants (Ids)", "Poteaux en intersection",
        "Poteaux coupés (u)", "Poteaux coupés (Ids)", "Poteaux coupants (u)", "Poteaux coupants (Ids)", "Volume", "Surface", "Matériau structurel",
    ],
    "Poutres": [
        "Id", "011EC_Lot", "012EC_Ouvrage", "013EC_Localisation", "014EC_Mode Constructif", "AI", "AS", "Hauteur totale", "Hauteur", "Sols en intersection", "Sols coupés (u)",
        "Sols coupés (Ids)", "Sols coupants (u)", "Sols coupants (Ids)", "Sol au-dessus", "Sol en-dessous", "Poteaux en intersection",
        "Poteaux coupés (u)", "Poteaux coupés (Ids)", "Poteaux coupants (u)", "Matériau structurel", "Elévation à la base", "Longueur de coupe",
    ],
    "Poteaux": [
        "Id", "011EC_Lot", "012EC_Ouvrage", "013EC_Localisation", "014EC_Mode Constructif", "Nom", "AI", "AS", "Hauteur", "Longueur",
        "Partie inférieure attachée", "Partie supérieure attachée", "Sols en intersection", "Sols coupés (u)", "Sols coupés (Ids)",
        "Sols coupants (u)", "Sols coupants (Ids)", "Poutres en intersection", "Poutres coupés (u)", "Poutres coupés (Ids)", "Poutres coupants (u)",
        "Poutres coupants (Ids)", "Matériau structurel", "Marque d'emplacement du poteau", "Décalage supérieur", "Décalage inférieur",
    ]
}

### ========================================
### Prefix mapping for different DataFrames
### ========================================
PREFIXES = {
    "Murs": "Mur_",
    "Sols": "Sol_",
    "Poutres": "Poutre_",
    "Poteaux": "Poteau_",
}

##################  CONSTANTS  #####################

################## VALIDATIONS #################

env_valid_options = dict(
)

def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)
