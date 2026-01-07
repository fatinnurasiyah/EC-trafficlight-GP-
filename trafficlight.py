import streamlit as st
import pandas as pd
import numpy as np
import random
import time

# =========================
# Page Config
# =========================
st.set_page_config(page_title="GP Traffic Light Optimization", layout="wide")

st.title("🚦 Traffic Light Optimization using Genetic Programming (GP)")
st.markdown("**Computational Evolution Case Study**")

# =========================
# Load Dataset
# =========================
st.subheader("📂 Traffic Dataset")

data = pd.read_csv("traffic_dataset.csv")

# =========================
# Encode categorical column
# =========================
if data["time_of_day"].dtype == object:
    data["time_of_day"] = data["time_of_day"].map({
        "morning": 0,
        "afternoon": 1,
        "evening": 2,
        "night": 3
    })

st.markdown("**Encoded Dataset Preview (After Preprocessing):**")
st.dataframe(data.head())

# =========================
# Features & Target
# =========================
X = data.drop(columns=["waiting_time"]).astype(float).values
y = data["waiting_time"].astype(float).values
feature_names = list(data.drop(columns=["waiting_time"]).columns)

# =========================
# Sidebar Parameters
# =========================
st.sidebar.header("⚙️ GP Parameters")

population_size = st.sidebar.slider("Population Size", 20, 100, 50)
generations = st.sidebar.slider("Generations", 5, 100, 20)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.50, 0.10)
coef_range = st.sidebar.slider("Coefficient Range (±)", 0.5, 5.0, 2.0)
bias_range = st.sidebar.slider("Bias Range (±)", 1.0, 10.0, 5.0)

# =========================
# GP Helper Functions
# =========================
def random_expression():
    feature = random.randint(0, len(feature_names) - 1)
    coef = random.uniform(-coef_range, coef_range)
    bias = random.uniform(-bias_range, bias_range)
    return (coef, feature, bias)

def predict(expr, X):
    coef, feature, bias = expr
    return coef * X[:, feature] + bias

def fitness(expr, X, y):
    y_pred = predict(expr, X)
    return np.mean((y - y_pred) ** 2)

def mutate(expr):
    coef, feature, bias = expr
    coef += random.uniform(-0.2 * coef_range, 0.2 * coef_range)
    bias += random.uniform(-0.2 * bias_range, 0.2 * bias_range)
    return (coef, feature, bias)

# =========================
# Run GP Optimization
# =========================
st.subheader("🚦 Optimization Results (Genetic Programming)")

if st.button("Run Genetic Programming (GP)"):

    start_time = time.time()

    with st.spinner("Running GP evolution..."):

        population = [random_expression() for _ in range(population_size)]
        fitness_history = []

        for gen in range(generations):
            scored = [(expr, fitness(expr, X, y)) for expr in population]
            scored.sort(key=lambda x: x[1])

            best_fitness_gen = scored[0][1]
            fitness_history.append(best_fitness_gen)

            # Selection (top 50%)
            population = [expr for expr, _ in scored[:population_size // 2]]

            # Reproduction
            while len(population) < population_size:
                parent = random.choice(population)
                if random.random() < mutation_rate:
                    population.append(mutate(parent))
                else:
                    population.append(parent)

        best_expr = min(population, key=lambda e: fitness(e, X, y))
        best_fitness = fitness(best_expr, X, y)

    exec_time = time.time() - start_time

    coef, feature, bias = best_expr
    feature_name = feature_names[feature]
    y_pred = predict(best_expr, X)

    st.success(" Optimization Completed")

    # =========================
    # Best Model
    # =========================
    st.markdown("###  Best Interpretable Mathematical Model")
    st.code(
        f"waiting_time = {coef:.3f} × {feature_name} + {bias:.3f}"
    )

    st.write(f"📉 **Best Fitness (MSE):** {best_fitness:.4f}")
    st.write(f"⏱ **Execution Time:** {exec_time:.4f} seconds")

    # =========================
    # Convergence Graph
    # =========================
    st.subheader("📈 Convergence Behaviour")
    st.line_chart(
        pd.DataFrame(
            {"Best Fitness (MSE)": fitness_history}
        )
    )

    # =========================
    # Actual vs Predicted
    # =========================
    st.subheader("📊 Actual vs Predicted Waiting Time")
    st.scatter_chart(
        pd.DataFrame({
            "Actual Waiting Time": y,
            "Predicted Waiting Time": y_pred
        })
    )

    # =========================
    # Performance Analysis
    # =========================
    st.subheader("Performance Analysis")
    st.markdown(
        "- **Convergence Rate:** Rapid fitness improvement during early generations\n"
        "- **Accuracy:** GP-generated equation predicts waiting time effectively\n"
        "- **Interpretability:** Simple linear expression is human-readable\n"
        "- **Efficiency:** Low computation time due to symbolic regression"
    )

    # =========================
    # Conclusion
    # =========================
    st.subheader("Conclusion")
    st.markdown(
        "This Streamlit-based Genetic Programming system demonstrates how evolutionary computation "
        "can automatically generate an interpretable mathematical model for predicting traffic "
        "waiting time using traffic dataset attributes."
    )

