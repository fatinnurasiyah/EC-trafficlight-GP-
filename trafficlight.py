import streamlit as st
import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="GP Traffic Light Optimization",
    layout="wide"
)

st.title("🚦 Traffic Light Optimization using Genetic Programming (GP)")
st.markdown("**Computational Evolution Case Study**")

# =========================
# Load Dataset
# =========================
st.subheader("📂 Traffic Dataset")

data = pd.read_csv("traffic_dataset.csv")
st.dataframe(data.head())

# =========================
# Features & Target
# =========================
X = data.drop(columns=["waiting_time"]).values
y = data["waiting_time"].values

# =========================
# Sidebar Parameters
# =========================
st.sidebar.header("⚙️ GP Parameters")

pop_size = st.sidebar.slider("Population Size", 100, 1000, 300, step=100)
generations = st.sidebar.slider("Generations", 10, 200, 50)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
max_depth = st.sidebar.slider("Max Tree Depth", 2, 8, 4)

# =========================
# GP Model
# =========================
st.subheader("🧠 Genetic Programming Model")

if st.button("▶ Run GP Optimization"):
    with st.spinner("Running GP evolution..."):

        gp = SymbolicRegressor(
            population_size=pop_size,
            generations=generations,
            p_crossover=0.7,
            p_subtree_mutation=mutation_rate,
            p_point_mutation=0.1,
            max_depth=max_depth,
            stopping_criteria=0.01,
            random_state=42
        )

        gp.fit(X, y)

    st.success("✅ GP Optimization Completed")

    # =========================
    # Results
    # =========================
    st.subheader("🏆 Best GP Expression")
    st.code(str(gp._program))

    st.subheader("📊 Fitness Score")
    st.write(f"Fitness Value: {gp.fitness_}")

    # =========================
    # Prediction Analysis
    # =========================
    y_pred = gp.predict(X)

    result_df = pd.DataFrame({
        "Actual Waiting Time": y,
        "Predicted Waiting Time": y_pred
    })

    st.subheader("📈 Actual vs Predicted Waiting Time")
    st.scatter_chart(result_df)

    # =========================
    # Performance Analysis
    # =========================
    st.subheader("📌 Performance Analysis")
    st.markdown(
        "- **Convergence Rate**: Fast improvement in early generations\n"
        "- **Prediction Accuracy**: GP captures nonlinear traffic behavior\n"
        "- **Expression Complexity**: Controlled by tree depth\n\n"
        "**Conclusion**:\n"
        "- GP successfully models waiting time using traffic variables\n"
        "- Suitable for traffic light optimization problems"
    )

st.markdown("---")
st.markdown(
    "Developed for **Evolutionary Computation – Genetic Programming Case Study**"
)
