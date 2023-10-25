import streamlit as st
import pandas as pd
import numpy as np
import pickle

model_path = "model.pkl"
scaler_path = "scaler.pkl"
transformer_path = "log_transform.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

with open(transformer_path, "rb") as f:
    transformer = pickle.load(f)

res_types = {
    "Amoled": 0,
    "HD": 1,
    "Super Retina": 2,
    "LCD": 3,
    "O-LED": 4,
    "TFT": 7,
}

proc_types = {
    "Apple Chips": 1,
    "Google Tensor": 2,
    "Qualcomm Snapdragon": 9,
    "Samsung Exynos": 10,
    "Mediatek Dimensity": 5,
    "Mediatek Helio": 6,
    "Mediatek": 4,
    "Intel": 3,
    "Unisoc": 11,
    "Others": 8,
}

proc_core_types = {
    "Single Core": 5,
    "Dual Core": 1,
    "Quad Core": 4,
    "Hexa Core": 2,
    "Octa Core": 3,
    "Deca Core": 0,
}

numeric_features = ["Price", "RAM", "Internal Storage", "Battery Capacity"]
categoric_features = ["Resolution Type", "Processor Type", "Processor Core"]
cols = numeric_features + categoric_features


def main():
    st.title("Smartphone Price Prediction with Gaussian Process Regression")

    ram = st.number_input("Enter RAM (GB): ", 4, 64)
    rom = st.number_input("Enter Internal Storage (GB): ", 32, 1024)
    battery = st.number_input("Enter Battery Capacity (mAh): ", 2000, 10000)

    col1, col2, col3 = st.columns(3)
    with col1:
        res_type = st.radio("Select Resolution Type:", list(res_types.keys()), key=1)

    with col2:
        proc_core = st.radio(
            "Select Processor Core:", list(proc_core_types.keys()), key=2
        )

    with col3:
        proc_type = st.radio("Select Processor Type:", list(proc_types.keys()), key=3)

    features = pd.DataFrame(
        [
            [
                np.nan,
                ram,
                rom,
                battery,
                res_types[res_type],
                proc_types[proc_type],
                proc_core_types[proc_core],
            ]
        ],
        columns=cols,
    )

    transformed = transformer.transform(features[numeric_features])
    categoric = features[categoric_features]
    for col in categoric.columns:
        transformed[col] = features[col]
    scaled = scaler.transform(transformed)
    scaled_feats = pd.DataFrame(scaled, columns=transformed.columns)

    price, std = model.predict(scaled_feats.drop("Price", axis=1), return_std=True)

    min_price = price[0] - 0.3 * std
    max_price = price[0] + 0.3 * std

    price = np.expm1(
        scaler.inverse_transform(np.array([[price[0], 0, 0, 0, 0, 0, 0]]))[0][0]
    )
    min_price = np.expm1(
        scaler.inverse_transform(np.array([[min_price[0], 0, 0, 0, 0, 0, 0]]))[0][0]
    )
    max_price = np.expm1(
        scaler.inverse_transform(np.array([[max_price[0], 0, 0, 0, 0, 0, 0]]))[0][0]
    )
    st.success(f"The price of the smartphone is ₹{price:.2f}/-")
    st.info(f"However the price can vary between ₹{min_price:.2f}/- and ₹{max_price:.2f}/-")


if __name__ == "__main__":
    main()
