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
    #print(sheet_concat.columns.tolist()[0:5])

    sheet_concat.columns = make_unique([clean_col(c) for c in sheet_concat.columns])
    #print(sheet_concat.columns.tolist()[0:5])

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
        pipleline_name:str,
        n_estimators = 5000,
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
        ('model', MultiOutputClassifier(RandomForestClassifier(n_estimators=n_estimators, random_state=42, verbose=1)))
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
    model_name = f'bimpredictapp/models/testing/trained_{pipleline_name}.pkl'
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

def load_prefit_model(model_pikle) -> Pipeline:

    #pipeline=pickle.load(open(model_pikle,'rb'))
    pipeline=joblib.load(model_pikle)

    return pipeline


def pred(df_test:pd.DataFrame,
         pipeline: Pipeline) -> pd.DataFrame:
    """
    Make a prediction using the latest trained model
    """

    if isinstance(pipeline, list):
        pipeline = pipeline[0]


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
                    pred_poteaux:pd.DataFrame,
                    old_file_df:pd.DataFrame):

    #making copies for each df
    pred_murs_copy = pred_murs.copy()
    pred_sols_copy = pred_sols.copy()
    pred_poutres_copy = pred_poutres.copy()
    pred_poteaux_copy = pred_poteaux.copy()

    input_filename = 'output.xlsx'
    filename = f'bimpredictapp/data/predicting_data/{input_filename}'

    #writing the dfs to sheets in one file

    if isinstance(old_file_df, list):
        old_file_df = old_file_df[0]

    columns_to_drop = ['011EC_Lot', '012EC_Ouvrage', '013EC_Localisation','014EC_Mode Constructif']

    old_murs_df = old_file_df['Murs']
    old_murs_df.drop(columns_to_drop, axis=1, inplace=True)

    old_sols_df = old_file_df['Sols']
    old_sols_df.drop(columns_to_drop, axis=1, inplace=True)

    old_poutres_df = old_file_df['Poutres']
    old_poutres_df.drop(columns_to_drop, axis=1, inplace=True)

    old_poteaux_df = old_file_df['Poteaux']
    old_poteaux_df.drop(columns_to_drop, axis=1, inplace=True)

    def merge_with_id(pred_df, input_df):
        if 'Id' not in input_df.columns:
            raise ValueError("Colonne 'Id' absente du fichier d'entrée.")
        if len(pred_df) != len(input_df):
            raise ValueError("Les DataFrames prédiction et entrée n'ont pas la même taille.")
        df = pred_df.copy()
        df.insert(0, 'Id', input_df['Id'].values)
        return df

    #remove old empty columns

    #merging pred with old
    pred_murs_copy_id = merge_with_id(pred_murs_copy,old_murs_df)
    final_df_murs = pd.merge(pred_murs_copy_id, old_murs_df, on='Id', how="outer")

    pred_sols_copy_id = merge_with_id(pred_sols_copy, old_sols_df)
    final_df_sols = pd.merge(pred_sols_copy_id, old_sols_df, on='Id', how="outer")

    pred_poutres_copy_id = merge_with_id(pred_poutres_copy, old_poutres_df)
    final_df_poutres = pd.merge(pred_poutres_copy_id, old_poutres_df, on='Id', how="outer")

    pred_poteaux_copy_id = merge_with_id(pred_poteaux_copy, old_poteaux_df)
    final_df_poteaux = pd.merge(pred_poteaux_copy_id, old_poteaux_df, on='Id', how="outer")


    with pd.ExcelWriter(filename) as writer:
        final_df_murs.to_excel(writer, sheet_name='Murs')
        final_df_sols.to_excel(writer, sheet_name='Sols')
        final_df_poutres.to_excel(writer, sheet_name='Poutres')
        final_df_poteaux.to_excel(writer, sheet_name='Poteaux')

    if not os.path.isfile(filename):
        raise ValueError(f"Error saving {filename}!")

    return f"The excel file {filename} has been saved successfuly."

if __name__ == '__main__':

    #testing files
    data_dir =  RAW_DATA_DIR ## Training data files
    test_file = TESTING_DATA_DIR # testing the prediction
    model_pikle_all = MODEL_TEST_DIR_COMB
    model_pikle_murs = MODEL_TEST_DIR_MURS
    model_pikle_sols = MODEL_TEST_DIR_SOLS


    if MODE == 'training':
        all_df = load_excel_files(data_dir, EXCEL_SHEETS)
        sheet_concat = concat_features(all_df)

        murs_concat = sheet_concat[0]
        sols_concat = sheet_concat[1]
        poutres_concat = sheet_concat[2]
        poteaux_concat = sheet_concat[3]

        X_murs, y_murs_multi = preprocess(murs_concat)
        X_sols, y_sols_multi = preprocess(sols_concat)
        X_poutres, y_poutres_multi = preprocess(poutres_concat)
        X_poteaux, y_poteaux_multi = preprocess(poteaux_concat)

        pipeline_murs, X_murs_test, y_murs_test = train(X_murs, y_murs_multi,'pipleline_murs', n_estimators = 1500)
        pipeline_sols, X_sols_test, y_sols_test = train(X_sols, y_sols_multi,'pipleline_sols', n_estimators = 1500)
        pipeline_poutres, X_poutres_test, y_poutres_test = train(X_poutres, y_poutres_multi, 'pipeline_poutres', n_estimators = 1500)
        pipeline_potreaux, X_poteaux_test, y_poteaux_test = train(X_poteaux, y_poteaux_multi,'pipeline_potreaux', n_estimators = 1500)

        evaluate(X_murs_test, y_murs_test, pipeline_murs)
        evaluate(X_sols_test, y_sols_test, pipeline_sols)
        evaluate(X_poutres_test, y_poutres_test, pipeline_poutres)
        evaluate(X_poteaux_test, y_poteaux_test, pipeline_potreaux)



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

        #loading pipeline(s)
        pipeline = load_prefit_model(model_pikle_all)

        #predicting target features
        murs_pred = pred(new_X1, pipeline)
        sols_pred = pred(new_X2, pipeline)
        poutres_pred = pred(new_X3, pipeline)
        poteaux_pred = pred(new_X4, pipeline)


        #merging with old file and writing a new excel file
        save_excel_file(murs_pred,sols_pred, poutres_pred, poteaux_pred, test_df)

    else:
        print('ERROR: mode shoud be declared as tringing or predicting in the param.py file')
