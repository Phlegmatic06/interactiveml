import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis

st.set_page_config(page_title="Interactive ML Visual Lab", page_icon="📊", layout="wide")

st.title("Interactive Machine Learning Visual Lab")
st.caption("Explore Factor Analysis and Hidden Markov Models by changing inputs and observing how the results update.")

tab_fa, tab_hmm = st.tabs(["Factor Analysis", "HMM Message Passing"])

# ---------------- Factor Analysis ----------------
with tab_fa:
    st.header("1. Factor Analysis on the Iris dataset")
    st.write(
        "Factor Analysis models correlated observed variables using a smaller set of latent factors. "
        "Choose the input features and number of factors, then inspect the loadings and factor-score plot."
    )

    iris = load_iris(as_frame=True)
    all_features = list(iris.feature_names)
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

    left, right = st.columns([1, 2])
    with left:
        features = st.multiselect(
            "Observed variables",
            all_features,
            default=all_features,
            help="Choose at least two features. The selected variables are standardized before fitting."
        )
        n_factors = st.slider(
            "Number of latent factors",
            min_value=1, max_value=4, value=2,
            help="The number of latent dimensions used to represent the observed variables."
        )
        standardize = st.checkbox("Standardize selected variables", value=True)
        show_species = st.checkbox("Color points by Iris species", value=True)

    if len(features) < 2:
        st.warning("Select at least two observed variables to fit Factor Analysis.")
    else:
        X = df[features].to_numpy(dtype=float)
        if standardize:
            X_fit = StandardScaler().fit_transform(X)
        else:
            X_fit = X

        actual_factors = min(n_factors, len(features), max(1, len(df) - 1))
        fa = FactorAnalysis(n_components=actual_factors, random_state=42)
        scores = fa.fit_transform(X_fit)
        loadings = fa.components_.T

        # Communality is the sum of squared loadings for each observed feature.
        communalities = np.sum(loadings ** 2, axis=1)
        loadings_df = pd.DataFrame(
            loadings,
            index=features,
            columns=[f"Factor {i+1}" for i in range(actual_factors)]
        )
        loadings_df["Communality (sum of squared loadings)"] = communalities

        with right:
            st.subheader("Factor loadings")
            st.dataframe(loadings_df.style.format("{:.3f}"), use_container_width=True)

            if actual_factors >= 2:
                plot_df = pd.DataFrame({
                    "Factor 1 score": scores[:, 0],
                    "Factor 2 score": scores[:, 1],
                    "Species": df["species"]
                })
                fig = px.scatter(
                    plot_df,
                    x="Factor 1 score",
                    y="Factor 2 score",
                    color="Species" if show_species else None,
                    title="Iris observations in latent-factor space",
                    hover_data=["Species"]
                )
                fig.update_layout(height=480, legend_title_text="Species")
                st.plotly_chart(fig, use_container_width=True)
            else:
                plot_df = pd.DataFrame({
                    "Observation": np.arange(1, len(scores) + 1),
                    "Factor 1 score": scores[:, 0],
                    "Species": df["species"]
                })
                fig = px.scatter(
                    plot_df, x="Observation", y="Factor 1 score",
                    color="Species" if show_species else None,
                    title="Observation scores on Factor 1"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Selected-variable correlation matrix")
        corr = pd.DataFrame(X, columns=features).corr()
        corr_fig = px.imshow(
            corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu",
            title="Pearson correlation between selected observed variables"
        )
        corr_fig.update_layout(height=420)
        st.plotly_chart(corr_fig, use_container_width=True)

        st.info(
            "Interpretation: factor loadings describe how strongly each observed variable is associated "
            "with each latent factor. Factor Analysis is not PCA; the model focuses on shared covariance. "
            "Communality here is the sum of squared loadings and is not a PCA explained-variance percentage."
        )

# ---------------- HMM ----------------
with tab_hmm:
    st.header("2. Message passing in a chain-structured Hidden Markov Model")
    st.write(
        "Edit the model probabilities and observation sequence. The forward-backward algorithm computes "
        "the marginal probability of each hidden state at every time step."
    )

    state_names = ["Healthy", "Unwell"]
    observation_names = ["Normal", "Mild", "High"]

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Initial state probabilities")
        pi_healthy = st.slider("P(Healthy at t=1)", 0.01, 0.99, 0.70, 0.01)
        pi = np.array([pi_healthy, 1.0 - pi_healthy])
        st.caption(f"P(Unwell at t=1) = {1.0-pi_healthy:.2f}")

        st.subheader("Transition probabilities")
        p_hh = st.slider("P(Healthy → Healthy)", 0.01, 0.99, 0.80, 0.01)
        p_uh = st.slider("P(Unwell → Healthy)", 0.01, 0.99, 0.30, 0.01)
        A = np.array([[p_hh, 1-p_hh], [p_uh, 1-p_uh]])
        st.dataframe(
            pd.DataFrame(A, index=["From Healthy", "From Unwell"],
                         columns=["To Healthy", "To Unwell"]).style.format("{:.2f}"),
            use_container_width=True
        )

    with col_b:
        st.subheader("Observation probabilities")
        st.caption("Each row is a hidden state; each row sums to 1. Adjust the first and third probability in each row; the Mild probability is the remainder.")
        h_normal = st.slider("P(Normal | Healthy)", 0.01, 0.98, 0.60, 0.01)
        h_high = st.slider("P(High | Healthy)", 0.01, float(max(0.02, 0.99-h_normal)), min(0.10, 0.99-h_normal), 0.01)
        u_normal = st.slider("P(Normal | Unwell)", 0.01, 0.98, 0.10, 0.01)
        u_high = st.slider("P(High | Unwell)", 0.01, float(max(0.02, 0.99-u_normal)), min(0.60, 0.99-u_normal), 0.01)
        B = np.array([
            [h_normal, 1-h_normal-h_high, h_high],
            [u_normal, 1-u_normal-u_high, u_high]
        ])
        st.dataframe(
            pd.DataFrame(B, index=state_names, columns=observation_names).style.format("{:.2f}"),
            use_container_width=True
        )

    st.subheader("Observed sequence")
    seq_labels = st.multiselect(
        "Select observations in chronological order (repeat observations if needed)",
        observation_names,
        default=["Normal", "Mild", "High", "Mild"]
    )
    st.caption("For a sequence with repeated observations, use the sequence editor below. It accepts comma-separated values.")
    seq_text = st.text_input("Sequence editor", value="Normal, Mild, High, Mild")
    obs_map = {name: i for i, name in enumerate(observation_names)}
    try:
        obs = [obs_map[x.strip().title()] for x in seq_text.split(",") if x.strip()]
        if not obs:
            raise ValueError("Sequence cannot be empty.")
    except (KeyError, ValueError):
        st.error("Use only Normal, Mild, and High, separated by commas.")
        obs = [0, 1, 2, 1]

    if np.any(B < 0):
        st.error("Observation probabilities became invalid. Adjust the sliders so the row probabilities remain valid.")
    else:
        T = len(obs)
        alpha = np.zeros((T, 2))
        beta = np.zeros((T, 2))
        alpha[0] = pi * B[:, obs[0]]
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ A) * B[:, obs[t]]
        sequence_prob = float(alpha[-1].sum())

        beta[-1] = 1.0
        for t in range(T-2, -1, -1):
            beta[t] = A @ (B[:, obs[t+1]] * beta[t+1])

        raw_marginals = alpha * beta
        marginals = raw_marginals / raw_marginals.sum(axis=1, keepdims=True)

        result_df = pd.DataFrame({
            "Time step": np.arange(1, T+1),
            "Observation": [observation_names[i] for i in obs],
            "P(Healthy | all observations)": marginals[:, 0],
            "P(Unwell | all observations)": marginals[:, 1]
        })
        metric1, metric2 = st.columns(2)
        metric1.metric("Sequence probability", f"{sequence_prob:.6f}")
        metric2.metric("Number of time steps", str(T))

        st.subheader("Marginal probabilities after forward-backward message passing")
        st.dataframe(result_df.style.format({
            "P(Healthy | all observations)": "{:.3f}",
            "P(Unwell | all observations)": "{:.3f}"
        }), use_container_width=True)

        chart_df = result_df.melt(
            id_vars=["Time step", "Observation"],
            value_vars=["P(Healthy | all observations)", "P(Unwell | all observations)"],
            var_name="Hidden state", value_name="Marginal probability"
        )
        chart_df["Hidden state"] = chart_df["Hidden state"].str.replace("P(", "").str.replace(" | all observations)", "", regex=False)
        fig = px.line(
            chart_df, x="Time step", y="Marginal probability", color="Hidden state",
            markers=True, hover_data=["Observation"],
            title="Hidden-state marginal probabilities by time step"
        )
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show forward (α) and backward (β) messages"):
            alpha_df = pd.DataFrame(alpha, columns=["Healthy", "Unwell"])
            alpha_df.insert(0, "Time step", np.arange(1, T+1))
            beta_df = pd.DataFrame(beta, columns=["Healthy", "Unwell"])
            beta_df.insert(0, "Time step", np.arange(1, T+1))
            c1, c2 = st.columns(2)
            with c1:
                st.write("Forward messages α")
                st.dataframe(alpha_df.style.format("{:.6f}"), use_container_width=True)
            with c2:
                st.write("Backward messages β")
                st.dataframe(beta_df.style.format("{:.6f}"), use_container_width=True)

        st.info(
            "Forward pass: αₜ(j) = emission(j, oₜ) × Σᵢ αₜ₋₁(i) transition(i, j). "
            "Backward pass: βₜ(i) = Σⱼ transition(i, j) emission(j, oₜ₊₁) βₜ₊₁(j). "
            "The marginal is proportional to αₜ(i)βₜ(i), normalized across states."
        )

st.divider()
st.caption("Educational demonstration. Factor Analysis uses the built-in Iris dataset; the HMM sequence is illustrative.")
