
import pandas as pd
import numpy as np

from bimpredictapp.params import *



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
