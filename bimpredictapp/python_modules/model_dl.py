import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# Define hyperparameter search space
param_space = {
    'units_1': Integer(64, 256), 'units_2': Integer(32, 128),
    'dropout_1': Real(0.1, 0.5), 'dropout_2': Real(0.1, 0.5),
    'learning_rate': Real(0.0001, 0.01, prior='log-uniform'), 'batch_size': Integer(16, 64)
}

# Build model dynamically
def build_model(units_1, units_2, dropout_1, dropout_2, learning_rate):
    model = keras.Sequential([
        keras.layers.Dense(units_1, activation='relu', input_shape=(X_train_combined.shape[1],)),
        keras.layers.Dropout(dropout_1),
        keras.layers.Dense(units_2, activation='relu'),
        keras.layers.Dropout(dropout_2),
        keras.layers.Dense(len(set(y_train)), activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Tune hyperparameters
opt = BayesSearchCV(
    estimator=keras.wrappers.scikit_learn.KerasClassifier(build_model),
    search_spaces=param_space, n_iter=50, cv=3, scoring='accuracy',
    n_jobs=-1, random_state=42
)
opt.fit(X_train_combined, y_train)
best_model, best_params = opt.best_estimator_, opt.best_params_

print(f"\n✅ Best Hyperparameters: {best_params}")

# Train final model
history = best_model.fit(X_train_combined, y_train, epochs=50, batch_size=best_params['batch_size'], validation_data=(X_test_combined, y_test))

# Plot learning curve
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title(f'Deep Learning Training Curve - {dataset_name}')
plt.legend()
plt.grid()
plt.savefig('plots/deep_learning_tuned_learning_curve.png')
plt.show()

# Save best model
best_model.model.save('models/deep_learning/best_model_tuned.keras')
print("\n🚀 Deep Learning Model Training & Hyperparameter Tuning Completed!")
