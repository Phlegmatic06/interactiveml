# Interactive ML Visual Lab

## Run locally

1. Install Python 3.10 or newer.
2. Extract this folder.
3. Open a terminal in the folder and run:

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

- Factor Analysis: choose Iris features, number of latent factors, standardization, and species coloring.
- HMM: edit initial probabilities, transition probabilities, emission probabilities, and the observed sequence. Forward/backward messages and state marginals update automatically.

The Iris section uses the scikit-learn Iris dataset. The HMM is an illustrative two-state, three-observation example, separate from Iris because Iris records are not a time sequence.
