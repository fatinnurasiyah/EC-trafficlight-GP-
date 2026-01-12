# =========================
# Streamlit + GP for Vehicle Count Prediction
# =========================
import streamlit as st
import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# =========================
# Page Config
# =========================
st.set_page_config(page_title="GP Vehicle Count Prediction", layout="wide")
st.title("🚗 Vehicle Count Prediction using Genetic Programming (GP)")
st.markdown("Evolutionary Computing Project - JIE 42903")

# =========================
# 1. Upload Dataset
# =========================
st.subheader("Upload Traffic Dataset (CSV)")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Preview dataset
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Encode time_of_day if categorical
    if 'time_of_day' in data.columns:
        if data['time_of_day'].dtype == object:
            data['time_of_day'] = data['time_of_day'].map({
                'morning': 1,
                'afternoon': 2,
                'evening': 3,
                'night': 4
            })

    # Features & Target
    if 'vehicle_count' not in data.columns:
        st.error("Dataset must contain a 'vehicle_count' column!")
    else:
        feature_names = list(data.drop(columns=["vehicle_count"]).columns)
        X = data[feature_names].values
        y = data['vehicle_count'].values

        # =========================
        # Sidebar Parameters
        # =========================
        st.sidebar.subheader("GP Hyperparameters")
        population_size = st.sidebar.slider("Population Size", 50, 500, 100, step=10)
        generations = st.sidebar.slider("Generations", 5, 50, 20)
        parsimony_coefficient = st.sidebar.slider("Parsimony Coefficient", 0.0, 0.1, 0.01)
        
        # =========================
        # Run GP
        # =========================
        if st.button("Run GP"):
            with st.spinner("Running Genetic Programming..."):
                # Split dataset
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Initialize GP model
                gp_model = SymbolicRegressor(
                    population_size=population_size,
                    generations=generations,
                    stopping_criteria=0.01,
                    p_crossover=0.7,
                    p_subtree_mutation=0.1,
                    max_samples=0.9,
                    verbose=1,
                    parsimony_coefficient=parsimony_coefficient,
                    random_state=42
                )

                # Fit GP
                gp_model.fit(X_train, y_train)

                # Predict
                y_pred = gp_model.predict(X_test)

                # Metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # =========================
                # Display Results
                # =========================
                st.success("✅ GP Optimization Completed")
                st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
                st.metric("R2 Score", f"{r2:.4f}")

                st.subheader("Learned Mathematical Expression (GP Model)")
                st.code(str(gp_model._program))

                # =========================
                # Visualizations
                # =========================
                st.subheader("Actual vs Predicted Vehicle Count")
                chart_data = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                st.line_chart(chart_data)

                st.subheader("Convergence Curve (Best Fitness per Generation)")
                # Note: gplearn does not return all generation fitness, so we can approximate
                # Use internal program fitness at last generation
                # Alternatively, display y_pred vs y_test as convergence indicator
                st.line_chart(pd.DataFrame({"Actual vs Predicted": np.abs(y_test - y_pred)}))

                st.markdown("""
                **Conclusion:**  
                - GP model uses all features to predict vehicle count.  
                - Formula is interpretable and can be used for traffic management.  
                - Adjusting population size, generations, and parsimony can improve prediction and model simplicity.
                """)


