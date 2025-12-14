import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import lightning.pytorch as pl
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# 0. Streamlit page setup
# -----------------------------
st.set_page_config(page_title="Drug Release Predictor", layout="wide")
st.title("Drug Release Predictor")

# -----------------------------
# 1. Define MLP classes (self-contained)
# -----------------------------
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class LitMLP(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.0, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLPModel(input_dim, hidden_dim, num_layers, dropout)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

# -----------------------------
# 2. Load saved scalers, feature columns, and checkpoint
# -----------------------------
scaler_X = joblib.load("scaler_X.save")
FEATURE_COLS = joblib.load("feature_cols.save")
model = LitMLP.load_from_checkpoint("mlp_final.ckpt", input_dim=len(FEATURE_COLS))
model.eval()

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.markdown("### Upload dataset with features to predict drug release")
uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("File uploaded successfully!")

    # Check required columns
    missing_cols = [c for c c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        # Scale and predict
        X_new = scaler_X.transform(df[FEATURE_COLS].values)
        X_new_tensor = torch.tensor(X_new, dtype=torch.float32)
        with torch.no_grad():
            preds = model(X_new_tensor).numpy().flatten()
        df['Predicted_Release'] = preds

        # -----------------------------
        # Tabs for UI
        # -----------------------------
        tab1, tab2, tab3 = st.tabs(["Data Preview", "Predictions", "Release Plot"])

        with tab1:
            st.write("Uploaded Data Preview:")
            st.dataframe(df)

        with tab2:
            st.write("Predicted Drug Release:")
            st.dataframe(df[['Predicted_Release'] + FEATURE_COLS])

            # Download button
            st.download_button(
                label="Download predictions CSV",
                data=df.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv"
            )

        with tab3:
            st.write("Predicted Drug Release Plot")
            plt.figure(figsize=(8,5))
            plt.plot(preds, marker='o', linestyle='-', color='b')
            plt.xlabel("Sample Index")
            plt.ylabel("Predicted Release")
            plt.title("Predicted Drug Release for Uploaded Samples")
            st.pyplot(plt)
