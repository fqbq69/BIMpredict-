import numpy as np

# Timing the TF import
#print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
#start = time.perf_counter()

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")


#### THIS IS THE TAXIFARE MODELS - WE ARE GOING TO CREATE OUR OWN

def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """

    print("✅ Model initialized")

    return model

def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """

    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=2,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history

def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(f"✅ Model evaluated, MAE: {round(mae, 2)}")

    return metrics
