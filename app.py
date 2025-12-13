import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import lightning.pytorch as pl
import joblib

st.title("Drug Release Predictor")

# -----------------------------
# Define MLP classes (self-contained)
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
        layers.append(nn.Linear(in_dim,1))
        self.net = nn.Sequential(*layers)
