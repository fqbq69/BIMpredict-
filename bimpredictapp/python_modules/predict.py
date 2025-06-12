
import pandas as pd
import numpy as np
import joblib

from bimpredictapp.params import *
from bimpredictapp.python_modules.encoders import load_encoders, encode_new_data



def make_ml_predictions(X_new):
    """Loads and applies all machine learning models."""
    feature_encoders, _ = load_encoders()
    X_encoded = encode_new_data(X_new.copy(), feature_encoders)

    if os.path.exists(MODEL_FEATURES_PATH):
        model_features = joblib.load(MODEL_FEATURES_PATH)
        X_encoded = X_encoded.reindex(columns=model_features, fill_value=0)

    models = {f.replace("_optimized.pkl", "").replace("_combined.pkl", ""): joblib.load(os.path.join(ML_MODELS_DIR, f))
              for f in os.listdir(ML_MODELS_DIR) if f.endswith(".pkl")}

    predictions = {name: model.predict(X_encoded) for name, model in models.items()}
    return max(predictions.items(), key=lambda x: np.mean(x[1]))  # Returns best ML prediction


def make_dl_predictions(X_new):
    """Loads and applies all deep learning models."""
    models = {f.replace("_best_model.keras", "").replace("_tuned.keras", ""): tf.keras.models.load_model(os.path.join(DL_MODELS_DIR, f))
              for f in os.listdir(DL_MODELS_DIR) if f.endswith(".keras")}

    predictions = {name: np.argmax(model.predict(X_new), axis=1) for name, model in models.items()}
    return max(predictions.items(), key=lambda x: np.mean(x[1]))


def predict_best_model(X_new):
    """Runs predictions across ML and DL models and picks the best one."""
    best_ml = make_ml_predictions(X_new)
    best_dl = make_dl_predictions(X_new)

    return best_ml if np.mean(best_ml[1]) > np.mean(best_dl[1]) else best_dl

def predict_all_models(X_new):
    """Runs predictions across ML and DL models and returns all results."""
    predictions = {"ML": {}, "Deep Learning": {}}

    # Machine Learning Predictions
    for name, model in make_ml_predictions(X_new).items():
        predictions["ML"][name] = model.predict(X_new)

    # Deep Learning Predictions
    for name, model in make_dl_predictions(X_new).items():
        predictions["Deep Learning"][name] = np.argmax(model.predict(X_new), axis=1)

    return predictions

# Generate model predictions
# predictions = predict_all_models(X_new)

# Run evaluation
# evaluation_results = evaluate_predictions(predictions, new_dataframes, detected_targets)

# Print final model rankings based on accuracy
#print("\n🚀 Final Model Evaluation Rankings:")

#rank models
def rank_models(evaluation_results):
    ranking_df = pd.DataFrame([
        (df_name, model_name, acc) for df_name, models in evaluation_results.items() for model_name, acc in models.items()
    ], columns=["Dataset", "Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)

    print(ranking_df.to_string(index=False))

#decode prediction
def decode_predictions(predictions, target_encoders):
    """Decodes encoded model predictions back to original text values."""
    decoded_results = {}

    for df_name, model_results in predictions.items():
        decoded_results[df_name] = {}

        # Decode ML model predictions
        for model_name, encoded_preds in model_results["ML"].items():
            decoded_results[df_name][model_name] = target_encoders[df_name].inverse_transform(encoded_preds)

        # Decode Deep Learning model predictions
        for model_name, encoded_preds in model_results["Deep Learning"].items():
            decoded_results[df_name][model_name] = target_encoders[df_name].inverse_transform(encoded_preds)

    return decoded_results
