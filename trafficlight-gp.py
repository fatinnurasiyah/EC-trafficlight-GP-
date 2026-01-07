import streamlit as st
import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.title("🚦 Traffic Light Optimization using Genetic Programming")

# Upload dataset
uploaded_file = st.file_uploader("Upload Traffic Dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Dataset Preview", df.head())

    # Features and target
    X = df[['vehicle_count', 'average_speed', 'lane_occupancy',
            'flow_rate', 'time_of_day']]
    y = df['waiting_time']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.subheader("🧬 Training Genetic Programming Model")

    gp = SymbolicRegressor(
        population_size=500,
        generations=20,
        stopping_criteria=0.01,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_depth=5,
        function_set=['add', 'sub', 'mul', 'div'],
        random_state=42,
        verbose=1
    )

    if st.button("Train Model"):
        gp.fit(X_train, y_train)

        # Prediction
        y_pred = gp.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        st.success("✅ Model Training Completed")

        st.subheader("📉 Model Performance")
        st.write("Mean Squared Error (MSE):", mse)

        st.subheader("📐 Generated Mathematical Model")
        st.code(str(gp._program))
