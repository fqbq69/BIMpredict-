import numpy as np
import pandas as pd
import os
from bimpredictapp.params import *

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Define ML models
# Run the pipeline
#X_combined, y_combined = process_data(final_cleaned_dataframes, TARGET_COLUMNS)
#train_models(X_combined, y_combined)

def initialize_simple_models():
    models = {
        "Logistic Regression": LogisticRegression(solver="saga", max_iter=5000, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
    }
    return models

def process_data(final_cleaned_dataframes, TARGET_COLUMNS) -> tuple:
    """Detects missing values, merges datasets, and resets index."""
    all_X, all_y = [], []
    print("\n✅ Checking available dataframes and target columns...")

    for df_name, df in final_cleaned_dataframes.items():
        existing_targets = [col for col in df.columns if any(target in col for target in TARGET_COLUMNS)]
        if not existing_targets:
            print(f"⚠️ {df_name}: No matching target columns found.")
            continue

        print(f"\n🔍 Processing {df_name} - Found target columns: {existing_targets}")

        for target_column in existing_targets:
            X, y = df.drop(columns=existing_targets), df[target_column]
            if y.nunique() == 1:
                print(f"⚠️ Skipping {df_name}_{target_column}: Only one class present.")
                continue

            all_X.append(X.reset_index(drop=True))
            all_y.append(y.reset_index(drop=True))

    if not all_X or not all_y:
        raise ValueError("🚨 No valid datasets found. Check TARGET_COLUMNS or ensure target values vary.")

    X_combined, y_combined = pd.concat(all_X, axis=0).reset_index(drop=True), pd.concat(all_y, axis=0).reset_index(drop=True)
    print(f"\n✅ Final merged dataset shape: {X_combined.shape}, {y_combined.shape}")
    return X_combined, y_combined

###############################################
# TRINGING AND SAVING MODELS
###############################################

def train_simple_models(X_combined, y_combined):
    """Trains multiple ML models & evaluates performance."""

    models = initialize_simple_models()

    failed_models = []

    print("\n🔍 Handling missing values...")
    X_combined = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X_combined), columns=X_combined.columns)
    y_combined.dropna(inplace=True)

    # Feature scaling
    scaler = StandardScaler()
    X_combined = pd.DataFrame(scaler.fit_transform(X_combined), columns=X_combined.columns)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

    model_results = {}

    plt.figure(figsize=(8, 5))
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
            model.fit(X_train, y_train)
            test_accuracy = accuracy_score(y_test, model.predict(X_test))
            model_results[name] = test_accuracy

            print(f"✅ {name}: Test Accuracy = {test_accuracy:.4f}")

            # Learning Curve
            train_sizes, train_scores, test_scores = learning_curve(
                model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 5)
            )
            plt.plot(train_sizes, np.mean(test_scores, axis=1), marker='o', label=f"{name} (Acc: {test_accuracy:.2f})")

        except Exception as e:
            print(f"⚠️ Error training {name}: {e}")
            failed_models.append(name)

    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve - All Models")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

    # Rank models
    ranked_models = sorted(model_results.items(), key=lambda x: x[1], reverse=True)
    print("\n📊 Model Rankings by Test Accuracy:")
    print(pd.DataFrame(ranked_models, columns=["Model", "Test Accuracy"]).to_string(index=False))

    # Save top models
    for name, _ in ranked_models[:2]:
        models[name].fit(X_combined, y_combined)
        joblib.dump(models[name], f'models/machine_learning/{name.replace(" ", "_")}_combined.pkl')

    print("\n🚀 Model evaluation, ranking, and saving completed!")
    print(f"⚠️ Models that failed: {failed_models}")


###############################################
# TRINGING AND SAVING HYPER MODELS
###############################################

# Define models and Bayesian hyperparameter search spaces
def initialize_hyper_models():
    models = {
        "Random Forest": (RandomForestClassifier(random_state=42), {
            'n_estimators': Integer(100, 1000), 'max_depth': Integer(3, 30),
            'min_samples_split': Integer(2, 15), 'max_features': Categorical(['sqrt', 'log2', None])
        }),
        "Logistic Regression": (LogisticRegression(max_iter=5000, random_state=42), {
            'C': Real(0.01, 10, prior='log-uniform'), 'solver': Categorical(['liblinear', 'lbfgs', 'saga'])
        }),
        "SVM": (SVC(probability=True, random_state=42), {
            'C': Real(0.1, 10, prior='log-uniform'), 'gamma': Real(0.01, 1, prior='log-uniform'),
            'kernel': Categorical(['linear', 'rbf'])
        }),
        "KNN": (KNeighborsClassifier(), {
            'n_neighbors': Integer(3, 15), 'weights': Categorical(['uniform', 'distance']),
            'metric': Categorical(['euclidean', 'manhattan', 'minkowski'])
        }),
        "Decision Tree": (DecisionTreeClassifier(random_state=42), {
            'max_depth': Integer(3, 20), 'min_samples_split': Integer(2, 10)
        }),
        "AdaBoost": (AdaBoostClassifier(random_state=42), {
            'n_estimators': Integer(50, 500), 'learning_rate': Real(0.01, 1, prior='log-uniform')
        }),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=42), {
            'n_estimators': Integer(50, 500), 'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'max_depth': Integer(3, 15)
        })
    }
    return models

def optimize_hyper_model(model, param_space, X_train, y_train):
    """Bayesian hyperparameter optimization."""
    opt = BayesSearchCV(model, param_space, n_iter=50, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
    opt.fit(X_train, y_train)
    return opt.best_estimator_, opt.best_score_

def train_and_rank_hyper_models(X_train, y_train):
    """Tunes models, ranks them, and saves the best two."""
    results = {}

    for name, (model, param_space) in models.items():
        print(f"\n🔍 Optimizing {name}...")
        try:
            best_model, best_score = optimize_model(model, param_space, X_train, y_train)
            results[name] = (best_model, best_score)
            print(f"✅ {name}: Accuracy = {best_score:.4f}")
        except Exception as e:
            print(f"⚠️ {name} failed: {e}")

    # Sort models by accuracy and save the top two
    top_models = sorted(results.items(), key=lambda x: x[1][1], reverse=True)[:2]
    for name, (model, _) in top_models:
        joblib.dump(model, f'models/machine_learning/{name.replace(" ", "_")}_optimized.pkl')

    print("\n🚀 Saved top 2 models:", [name for name, _ in top_models])
