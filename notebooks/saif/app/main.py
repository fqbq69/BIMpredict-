import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import io

# Set app title and description
st.title('🏗️ BIM Wall Property Predictor')
st.markdown("""
Predict thermal properties of construction walls based on BIM data.
Upload your Excel file to get predictions for the **Coefficient de transfert thermique (U)**.
""")

# File upload section
st.header('1. Upload Data')
uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])

def load_data(file):
    """Load and preprocess the Excel data"""
    df = pd.read_excel(file, sheet_name='Murs')

    # Preprocessing
    df = df.replace('<Aucun>', np.nan)
    df = df.dropna(axis=1, how='all')

    # Feature selection
    features = ['Hauteur', 'Epaisseur', 'Volume', 'Surface',
                'Rugosité', 'Coefficient d\'absorbance', 'Masse thermique']

    # Target variable
    target = 'Coefficient de transfert thermique (U)'

    return df, features, target

if uploaded_file:
    # Load and show data
    df, features, target = load_data(uploaded_file)
    st.success('Data successfully loaded!')

    # Show preview
    st.subheader('Data Preview')
    st.write(df[features + [target]].head())

    # Check for target column
    if target not in df.columns:
        st.error(f"Target column '{target}' not found in the uploaded file!")
        st.stop()

    # Model training section
    st.header('2. Train Prediction Model')

    # Handle missing values
    df_train = df.dropna(subset=[target] + features)

    if df_train.empty:
        st.error("No valid data for training. Check missing values!")
        st.stop()

    X = df_train[features]
    y = df_train[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.success(f"Model trained! R²: {r2:.3f}, MAE: {mae:.4f}")

    # Prediction section
    st.header('3. Generate Predictions')

    # Predict for all rows with features
    valid_rows = df[features].notna().all(axis=1)
    df.loc[valid_rows, 'Predicted U'] = model.predict(df.loc[valid_rows, features])

    # Show results
    st.subheader('Prediction Results')
    st.dataframe(df[['Nom', target, 'Predicted U']].head(10))

    # Visualization
    st.subheader('Actual vs Predicted Values')
    chart_data = pd.DataFrame({
        'Actual': df[target].dropna(),
        'Predicted': df.loc[df[target].notna(), 'Predicted U']
    })
    st.scatter_chart(chart_data)

    # Download results
    st.header('4. Download Results')

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Predictions', index=False)

    st.download_button(
        label="Download Predictions",
        data=output.getvalue(),
        file_name='predictions.xlsx',
        mime='application/vnd.ms-excel'
    )

else:
    st.info("Please upload an Excel file to get started")
