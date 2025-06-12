import numpy as np
import pandas as pd
from colorama import Fore, Style
import joblib

import os
from os import listdir
from os.path import isfile, join

import tensorflow as tf
from tensorflow import keras
import pickle

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

from bimpredictapp.params import *

from bimpredictapp.python_modules.load import load_excel_file
from bimpredictapp.python_modules.predict import predict_best_model
from bimpredictapp.python_modules.data import clean_col,make_unique



def load_excel_files(excel_folder,sheets) -> list: #returns a list of dicts

    files = [join(excel_folder, x) for x in listdir(excel_folder) if isfile(join(excel_folder, x))]

    all_data_list = []
    for file in files:
        print(f"loading sheets from {file}.")
        df_dict = load_excel_file(file,sheets)
        all_data_list.append(df_dict)

    return all_data_list

def concat_features(list_of_dict): #[{murs:mudrs_df, },{etc..}..]

    murs_df = []
    sols_df = []
    poutres_df = []
    poteaux_df = []

    for df_dict in list_of_dict:
        murs_df.append(df_dict['Murs'])
        sols_df.append(df_dict['Sols'])
        poutres_df.append(df_dict['Poutres'])
        poteaux_df.append(df_dict['Poteaux'])


    if murs_df:
        murs_concat = pd.concat(murs_df, ignore_index=True)
        print(f"Total concaténé : {murs_concat.shape[0]} lignes, {murs_concat.shape[1]} colonnes")
    else:
        print("Aucun mur trouvé.")
    if sols_df:
        sols_concat = pd.concat(sols_df, ignore_index=True)
        print(f"Total concaténé : {sols_concat.shape[0]} lignes, {sols_concat.shape[1]} colonnes")
    else:
        print("Aucun sol trouvé.")
    if poutres_df:
        poutres_concat = pd.concat(poutres_df, ignore_index=True)
        print(f"Total concaténé : {poutres_concat.shape[0]} lignes, {poutres_concat.shape[1]} colonnes")
    else:
        print("Aucun poutre trouvé.")
    if poteaux_df:
        poteaux_concat = pd.concat(poteaux_df, ignore_index=True)
        print(f"Total concaténé : {poteaux_concat.shape[0]} lignes, {poteaux_concat.shape[1]} colonnes")
    else:
        print("Aucun poteau trouvé.")

    return [murs_concat, sols_concat,poutres_concat,poteaux_concat]

### Preprocess the data
def preprocess(sheet_concat) -> tuple:
    """
    Prepares the data from each sheet and calls the required functions
    before training or using the models in the predicting process.
    """
    # Nettoyer les noms de colonnes et les rendre uniques

    sheet_concat.columns = [clean_col(c) for c in sheet_concat.columns]
    print(sheet_concat.columns.tolist()[0:5])

    sheet_concat.columns = make_unique([clean_col(c) for c in sheet_concat.columns])
    print(sheet_concat.columns.tolist()[0:5])

    # Déterminer les colonnes cibles effectivement présentes dans le DataFrame
    targets_in_df = [col for col in TARGET_FEATURES if col in sheet_concat.columns]

    if not targets_in_df:
        raise ValueError(f"Aucune colonne cible trouvée dans murs_concat. Colonnes disponibles :{sheet_concat.columns.tolist()}")

    # X et y_multi (après avoir rendu les colonnes uniques)
    X = sheet_concat.drop(columns=targets_in_df)
    y_multi = sheet_concat[targets_in_df]

    if X.shape[1] == 0:
        raise ValueError("Aucune variable explicative disponible après suppression des cibles. Vérifiez vos colonnes.")

    return X, y_multi

### Train the model(s)
def train( X: pd.DataFrame,
          y_multi:pd.DataFrame,
        split_ratio: float = 0.02, # 0.02 represents ~ 1 month of validation data on a 2009-2015 train set
        learning_rate=0.0005,
        batch_size = 256,
        patience = 2
    ) -> Pipeline:

    """
    Splitting the data into train/val/test
    Trainging the models if rewuired, with the sheets.
    Saves the model after training
    moves the last model into stagin
    """
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Convertir toutes les colonnes catégorielles en string pour éviter les erreurs d'encodage
    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].astype(str)

    # Supprimer les colonnes numériques entièrement vides
    num_features = [col for col in num_features if not X[col].isnull().all()]
    X = X[num_features + cat_features]

    # Pipeline de prétraitement
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), num_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_features)
        ]
    )

    # Pipeline complet avec MultiOutputClassifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', MultiOutputClassifier(RandomForestClassifier(n_estimators=5000, random_state=42, verbose=1)))
    ])

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y_multi,
                                                        test_size=0.2,
                                                        random_state=42)

    # Supprimer les lignes avec NaN dans les targets (train et test)
    train_notna = y_train.notna().all(axis=1)
    test_notna = y_test.notna().all(axis=1)
    X_train, y_train = X_train[train_notna], y_train[train_notna]
    X_test, y_test = X_test[test_notna], y_test[test_notna]

    # Entraînement
    pipeline.fit(X_train, y_train)

    # Prédiction et score baseline
    #joblib.dump(pipeline, 'randomforestmurspipeline.pkl')
    model_name = 'bimpredictapp/models/testing/trained_model.pkl'
    pickle.dump(pipeline, open(model_name, 'wb'))

    #print("Pipeline complet sauvegardé dans bimpredict_pipeline.pkl")

    return pipeline, X_test, y_test

### Evaluate Models
def evaluate(X_test: pd.DataFrame,
            y_test:pd.DataFrame,
             pipeline: Pipeline
            ) -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return MAE as a float
    """
    print("Accuracy calculée sur", len(y_test), "échantillons.")


    # Générer les prédictions sur le jeu de test
    y_pred = pipeline.predict(X_test)

    # Calculer le F1-score pour chaque colonne (target) séparément
    f1_micro_list = []
    f1_macro_list = []
    for i, col in enumerate(y_test.columns):
        f1_micro_list.append(f1_score(y_test.iloc[:, i], y_pred[:, i], average='micro'))
        f1_macro_list.append(f1_score(y_test.iloc[:, i], y_pred[:, i], average='macro'))

    # Moyenne des scores F1 sur toutes les cibles
    f1_micro_mean = np.mean(f1_micro_list)
    f1_macro_mean = np.mean(f1_macro_list)

    print(f"F1 micro (moyenne par cible): {f1_micro_mean:.4f}")
    print(f"F1 macro (moyenne par cible): {f1_macro_mean:.4f}")

    pass #returning eval values


### Load and predict

def pred(df_test:pd.DataFrame,
         pipeline: Pipeline) -> pd.DataFrame:
    """
    Make a prediction using the latest trained model
    """

    # Les targets à prédire (après nettoyage)
    targets = TARGET_FEATURES

    # Colonnes explicatives attendues par le pipeline
    features = pipeline.named_steps['preprocessor'].feature_names_in_

    # S'assurer que toutes les colonnes sont présentes
    for col in features:
        if col not in df_test.columns:
            df_test[col] = np.nan
            print(f"Colonne manquante ajoutée : {col}")

    X_test = df_test[features].copy()

    y_pred = pipeline.predict(X_test)

    y_pred_df = pd.DataFrame(y_pred, columns=targets)

    print(y_pred_df['013ec_localisation'].value_counts())

    # Mettre les prédictions dans un DataFrame
    df_pred = pd.DataFrame(y_pred, columns=targets)

    # Afficher les premières lignes
    print(df_pred.head())

    return  df_pred

def save_excel_file(pred_murs:pd.DataFrame,
                    pred_sols:pd.DataFrame,
                    pred_poutres:pd.DataFrame,
                    pred_poteaux:pd.DataFrame):

    #making copies for each df
    df1 = pred_murs.copy()
    df2 = pred_sols.copy()
    df3 = pred_poutres.copy()
    df4 = pred_poteaux.copy()

    filename = 'bimpredictapp/data/predicting_data/ouput.xlsx'
    #writing the dfs to sheets in one file

    with pd.ExcelWriter(filename) as writer:
        df1.to_excel(writer, sheet_name='Murs')
        df2.to_excel(writer, sheet_name='Sols')
        df3.to_excel(writer, sheet_name='Poutres')
        df4.to_excel(writer, sheet_name='Poteaux')

    if not os.path.isfile(filename):
        raise ValueError(f"Error saving {filename}!")

    return f"The excel file {filename} has been saved successfuly."

if __name__ == '__main__':

    #testing files
    data_dir =  RAW_DATA_DIR ## Training data files
    test_file = TESTING_DATA_DIR # testing the prediction
    model_pikle = MODEL_TEST_DIR
    temp_model = ""

    if MODE == 'training':
        all_df = load_excel_files(data_dir, EXCEL_SHEETS)
        sheet_concat = concat_features(all_df)
        murs_concat = sheet_concat[0]
        X, y_multi = preprocess(murs_concat)
        pipeline, X_test, y_test = train(X, y_multi)
        evaluate(X_test, y_test, pipeline)


    elif MODE == 'predicting':
        print("Let's predict new features...")

        #loading sheets as df and concating them

        test_df = load_excel_files(test_file,EXCEL_SHEETS)
        test_sheet_concat = concat_features(test_df)

        test_murs_concat = test_sheet_concat[0]
        test_sols_concat = test_sheet_concat[1]
        test_poutres_concat = test_sheet_concat[2]
        test_poteaux_concat = test_sheet_concat[3]

        #preproc each sheet df
        new_X1, new_y_multi1 = preprocess(test_murs_concat)
        new_X2, new_y_multi2 = preprocess(test_sols_concat)
        new_X3, new_y_multi3 = preprocess(test_poutres_concat)
        new_X4, new_y_multi4 = preprocess(test_poteaux_concat)

        #loading a model
        pipeline = load_prefit_model(model_pikle)

        #predicting target features
        murs_pred = pred(new_X1, pipeline)
        sols_pred = pred(new_X2, pipeline)
        poutres_pred = pred(new_X3, pipeline)
        poteaux_pred = pred(new_X4, pipeline)

        #writing an excel file
        save_excel_file(murs_pred,sols_pred, poutres_pred, poteaux_pred)

    else:
        print('ERROR: mode shoud be declared as tringing or predicting in the param.py file')
