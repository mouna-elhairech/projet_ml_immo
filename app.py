import os
import subprocess
import sys

import joblib
import numpy as np
import streamlit as st


@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        subprocess.run([sys.executable, "src/train.py"], check=True)
    return joblib.load("model.pkl")


model = load_model()

st.title("Estimation Immobilière")

med_inc = st.number_input("Revenu médian", value=8.3)

if st.button("Estimer"):
    features = np.array(
        [[med_inc, 41.0, 6.98, 1.02, 322.0, 2.55, 37.88, -122.23]]
    )
    prix = model.predict(features)[0] * 100000
    st.success(f"Prix estimé : {prix:.2f} $")
