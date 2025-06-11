import pandas as pd
import numpy as np
import os
import re
from bimpredictapp.params import *

from IPython.display import display, HTML
import missingno as msno
import matplotlib.pyplot as plt

def clean_columns(dataframes) -> pd.DataFrame:
    required_columns = {
        "Murs": ["Id", "011EC_Lot", "012EC_Ouvrage", "013EC_Localisation", "014EC_Mode Constructif", "Hauteur",
                "Epaisseur", "AI", "AS", "Sols en intersection", "Sols coupés (u)", "Sols coupés (Ids)",
                "Sols coupants (u)", "Sols coupants (Ids)", "Sol au-dessus", "Sol en-dessous", "Fenêtres", "Portes",
                "Ouvertures", "Murs imbriqués", "Mur multicouche", "Mur empilé", "Profil modifié", "Extension inférieure",
                "Extension supérieure", "Partie inférieure attachée", "Partie supérieure attachée", "Décalage supérieur",
                "Décalage inférieur", "Matériau structurel"],

        "Sols": ["Id", "011EC_Lot", "012EC_Ouvrage", "013EC_Localisation", "014EC_Mode Constructif", "Murs en intersection",
                "Murs coupés (u)", "Murs coupés (Ids)", "Murs coupants (u)", "Murs coupants (Ids)", "Poutres en intersection",
                "Poutres coupés (u)", "Poutres coupés (Ids)", "Poutres coupants (u)", "Poutres coupants (Ids)",
                "Poteaux en intersection", "Poteaux coupés (u)", "Poteaux coupés (Ids)", "Poteaux coupants (u)",
                "Poteaux coupants (Ids)", "Ouvertures", "Sol multicouche", "Profil modifié", "Décalage par rapport au niveau",
                "Epaisseur", "Lié au volume", "Etude de l'élévation à la base", "Etude de l'élévation en haut",
                "Epaisseur du porteur", "Elévation au niveau du noyau inférieur", "Elévation au niveau du noyau supérieur",
                "Elévation en haut", "Elévation à la base", "Matériau structurel"],

        "Poutres": ["Id", "011EC_Lot", "012EC_Ouvrage", "013EC_Localisation", "014EC_Mode Constructif", "AI", "AS",
                    "Hauteur totale", "Hauteur", "Sols en intersection", "Sols coupés (u)", "Sols coupés (Ids)",
                    "Sols coupants (u)", "Sols coupants (Ids)", "Sol au-dessus", "Sol en-dessous", "Poteaux en intersection",
                    "Poteaux coupés (u)", "Poteaux coupés (Ids)", "Poteaux coupants (u)", "Poteaux coupants (Ids)",
                    "Etat de la jonction", "Valeur de décalage Z", "Justification Z", "Valeur de décalage Y", "Justification Y",
                    "Justification YZ", "Matériau structurel", "Elévation du niveau de référence", "Elévation en haut",
                    "Rotation de la section", "Orientation", "Décalage du niveau d'arrivée", "Décalage du niveau de départ",
                    "Elévation à la base", "Longueur de coupe", "Longueur", "hauteur_section", "largeur_section"],

        "Poteaux": ["Id", "011EC_Lot", "012EC_Ouvrage", "013EC_Localisation", "014EC_Mode Constructif", "AI", "AS",
                    "Hauteur", "Longueur", "Partie inférieure attachée", "Partie supérieure attachée", "Sols en intersection",
                    "Sols coupés (u)", "Sols coupés (Ids)", "Sols coupants (u)", "Sols coupants (Ids)", "Poutres en intersection",
                    "Poutres coupés (u)", "Poutres coupés (Ids)", "Poutres coupants (u)", "Poutres coupants (Ids)",
                    "Matériau structurel", "Décalage supérieur", "Décalage inférieur", "Diamètre poteau", "h", "b",
                    "hauteur_section", "largeur_section"]
    }

    # Filter multiple dataframes dynamically
    cleaned_dataframes = {}  # Store cleaned versions

    for df_name, df in dataframes.items():
        print(f"\n🟢 Original shape of {df_name}: {df.shape}")

        # Automatically detect the correct category for filtering
        for category, columns in required_columns.items():
            if category.lower() in df_name.lower():  # Match dynamically
                try:
                    filtered_df = df[columns]  # Keep only the required columns
                except KeyError as e:
                    missing_columns = set(columns) - set(df.columns)
                    print(f"⚠️ Missing columns in {df_name}: {missing_columns}. Skipping this dataframe.")
                    continue
                cleaned_dataframes[df_name] = filtered_df
                print(f"✅ Shape after filtering {df_name}: {filtered_df.shape}")
                break  # Stop looping once the correct match is found
        else:
            print(f"⚠️ No matching category for {df_name}, skipping filtering.")

    # Add prefixes to column names based on the dataframe category and update index
    for name, df in cleaned_dataframes.items():
        if "murs" in name.lower():
            prefix = "murs_"
        elif "sols" in name.lower():
            prefix = "sols_"
        elif "poutres" in name.lower():
            prefix = "poutres_"
        elif "poteaux" in name.lower():
            prefix = "poteaux_"
        else:
            prefix = ""

        # Rename columns with the prefix
        df.rename(columns=lambda col: f"{prefix}{col}" if col.lower() != "id" else f"{prefix}id", inplace=True)

        # Drop the existing index and set the prefixed ID column as the new index
        id_column = f"{prefix}id"
        if id_column in df.columns:
            df.set_index(id_column, inplace=True)
            print(f"✅ Set '{id_column}' as index for {name}.")
        else:
            print(f"⚠️ '{id_column}' column not found in {name}, skipping index setting.")

        # Update the cleaned_dataframes dictionary
        cleaned_dataframes[name] = df

    return cleaned_dataframes

########################################
# MAPPING FEATURES
########################################

#mapped_dataframes = map_feature_names(cleaned_dataframes, required_columns)
def map_feature_names(cleaned_dataframes, required_columns) -> pd.DataFrame:
    """Maps cleaned dataframe column names to match required training feature names."""
    mapped_dataframes = {}

    for df_name, df in cleaned_dataframes.items():
        for category, expected_columns in required_columns.items():
            if category.lower() in df_name.lower():  # Match dynamically
                # Create mapping: {cleaned_col_name: expected_col_name}
                col_mapping = {cleaned_col: expected_col for cleaned_col in df.columns for expected_col in expected_columns if cleaned_col.lower() == expected_col.lower()}

                # Apply mapping to rename columns
                df_mapped = df.rename(columns=col_mapping)

                print(f"✅ Feature names mapped for {df_name}")
                mapped_dataframes[df_name] = df_mapped
                break  # Stop looping once category is matched

    return mapped_dataframes


########################################
# CLEAN COLUMN NAMES
########################################

def clean_column_names(df) -> pd.DataFrame:
    # Ensure all column names are lowercase, replace spaces with underscores, and remove special characters
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w_]", "", regex=True)
    )
    return df

# Clean column names in all provided DataFrames

def clean_all_column_names(df):

    cleaned_dataframes = {name: clean_column_names(df) for name, df in cleaned_dataframes.items()}
    print("✅ Column names cleaned successfully across all cleaned dataframes!")

    TARGET_COLUMNS = ['011ec_lot', '012ec_ouvrage', '013ec_localisation', '014ec_mode_constructif']
    final_cleaned_dataframes = {}
    target_columns_found = set()
    exception_keywords = ["coupés", "coupants", "011ec_lot", "012ec_ouvrage", "013ec_localisation", "014ec_mode_constructif"]

    for df_name, df in cleaned_dataframes.items():
        print(f"\n🟢 Processing {df_name}...")
        df = df.copy()
        initial_shape = df.shape
        print(f"📌 Initial shape: {initial_shape}")

        # Remove duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates:
            print(f"⚠️ Found {duplicates} duplicate rows. Removing...")
            df.drop_duplicates(inplace=True)
        else:
            print("✅ No duplicate rows found.")

        # Drop columns that are 100% missing unless they match exception keywords
        missing_cols = df.columns[df.isnull().mean() == 1]
        cols_to_drop = [col for col in missing_cols if not any(keyword in col.lower() for keyword in exception_keywords)]
        if cols_to_drop:
            print(f"⚠️ Dropping {len(cols_to_drop)} completely empty columns: {cols_to_drop}")
            df.drop(columns=cols_to_drop, inplace=True)
        else:
            print("✅ No fully missing columns detected (or all are exceptions).")

        mid_shape = df.shape

        # Ensure each target column exists, adding it with NaNs if missing (with naming policy)
        for target in TARGET_COLUMNS:
            target_col = f"{df_name.split('_')[-1].lower()}_{target.lower()}"
            if target_col not in df.columns:
                print(f"⚠️ Target column '{target_col}' missing in '{df_name}'. Adding it.")
                df[target_col] = float('nan')

        final_shape = df.shape
        if mid_shape != final_shape:
            print(f"📊 Shape adjustment: before {mid_shape}, after {final_shape}")

        # List and accumulate target columns found in the current DataFrame
        target_cols_in_df = [col for col in df.columns if any(t.lower() in col.lower() for t in TARGET_COLUMNS)]
        print(f"🎯 Target columns in '{df_name}': {target_cols_in_df}")
        target_columns_found.update(target_cols_in_df)

        final_cleaned_dataframes[df_name] = df
        print(f"📌 Final shape after cleaning: {final_shape}")

    print(f"\nTarget columns detected across datasets: {target_columns_found}")
    return final_cleaned_dataframes

# Ensure missing values are filled in the processed datasets unless in TARGET_COLUMNS
def check_missing_values(final_cleaned_dataframes) -> pd.DataFrame:
    for df_name, df in final_cleaned_dataframes.items():
        print(f"\n🟢 Filling missing values for {df_name}...")

        # Display shape before filling missing values
        initial_shape = df.shape
        print(f"📌 Initial shape before filling NaN: {initial_shape}")

        # Fill missing values with 0 for non-target columns
        non_target_columns = [col for col in df.columns if col not in TARGET_COLUMNS]
        df[non_target_columns] = df[non_target_columns].fillna(0)

        # Store updated dataframe back
        final_cleaned_dataframes[df_name] = df

        # Display shape after processing
        final_shape = df.shape
        print(f"✅ Final shape after filling NaN: {final_shape}")

    print("🚀 Missing values successfully handled across all datasets!")

    return final_cleaned_dataframes

def identify_target(final_cleaned_dataframes) -> set:
# Identify target columns dynamically across all DataFrames
    target_columns_found = set()
    for df_name, df in final_cleaned_dataframes.items():
        found_targets = [
            col for col in df.columns
            if any(target.lower() in col.lower() for target in TARGET_COLUMNS)
        ]
        target_columns_found.update(found_targets)

    print(f"\nTarget columns detected across datasets: {target_columns_found}")

    return target_columns_found
