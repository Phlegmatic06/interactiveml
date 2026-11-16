# Interactive ML Visual Lab

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder and run:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements-local.txt
streamlit run app.py
```

Streamlit prints a local URL (usually http://localhost:8501). Open it in your browser.

## What's interactive?

- **Factor Analysis**: choose Iris features, number of latent factors, standardization, and species coloring.
- **HMM Message Passing**: edit initial probabilities, transition probabilities, emission probabilities, and the observed sequence. Forward/backward messages and state marginals update automatically.
- **Adaptive Basis Function Models for Nonlinear Function Approximation**:
  - Based on the comparative research of:
    1. **Polynomial Regression (Fixed Global Basis)**
    2. **Fixed Radial Basis Function (RBF) Network (Uniform Rigid Grid)**
    3. **Adaptive Radial Basis Function (RBF) Network (Differentiable Centers $\mu_k$ and Widths $\sigma_k$ learned via Adam)**
  - Interactive target functions: Complex nonlinear benchmark, sharp Gaussian peak, Runge's polynomial stress test, and multi-frequency chirp.
  - Interactive parameter controls: sample size $N$, noise level $\sigma_\epsilon$, polynomial degree, center count $K$, bandwidth $\sigma$, training epochs, and learning rate.
  - Interactive Plotly visualizations:
    - Model approximation curves overlaid on ground truth and test points.
    - Side-by-side **Basis Function Decomposition** showing how Gaussian bells shift their centers and adjust widths to fit high-curvature regions.
    - Training loss convergence curve (MSE vs. Epoch).
    - Benchmark performance bar charts (RMSE & $R^2$).
    - Learned parameters inspection table.
    - Presentation-ready talking points and theoretical takeaways.
