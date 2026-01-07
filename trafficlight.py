import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gp_model import run_gp

st.set_page_config(page_title="Genetic Programming - Traffic Prediction", layout="wide")

st.title("🚦 Genetic Programming for Traffic Prediction")

# Sidebar
st.sidebar.header("GP Parameters")
pop_size = st.sidebar.slider("Population Size", 50, 500, 200)
ngen = st.sidebar.slider("Generations", 10, 100, 40)
cxpb = st.sidebar.slider("Crossover Probability", 0.1, 1.0, 0.7)
mutpb = st.sidebar.slider("Mutation Probability", 0.01, 0.5, 0.2)
max_depth = st.sidebar.slider("Max Tree Depth", 2, 6, 4)

# Load dataset
st.subheader("Traffic Dataset")
uploaded_file = st.file_uploader("Upload traffic CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    if st.button("Run Genetic Programming"):
        with st.spinner("Running Genetic Programming..."):
            best_ind, fitness_history = run_gp(
                X, y, pop_size, ngen, cxpb, mutpb, max_depth
            )

        st.success("GP Completed Successfully!")

        # Convergence plot
        st.subheader("📈 Convergence Curve")
        fig, ax = plt.subplots()
        ax.plot(fitness_history)
        ax.set_xlabel("Generation")
        ax.set_ylabel("MSE")
        st.pyplot(fig)

        # Best expression
        st.subheader("🌳 Best Evolved Expression")
        st.code(str(best_ind))

else:
    st.info("Please upload a traffic dataset CSV file.")
