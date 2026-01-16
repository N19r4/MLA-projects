# streamlit run app.py

import streamlit as st
import torch
import numpy as np
from PIL import Image
from monai.networks.nets.unet import UNet
from monai.transforms.compose import Compose
from monai.transforms.intensity.array import ScaleIntensity
from monai.transforms.spatial.array import Resize
from monai.transforms.utility.array import EnsureChannelFirst
import matplotlib.pyplot as plt

st.set_page_config(page_title="HCI - Segmentacja Raka Płuc", layout="wide")

st.title("System wspomagania diagnostyki raka płuc")

st.sidebar.image("media/ps-logo.png", width=150)
st.sidebar.header("Zespół Projektowy")
st.sidebar.write("**Piotr Skowroński** (Lider/LaTeX)")
st.sidebar.write("**Krzysztof Czuba** (Inżynier ML)")
st.sidebar.write("**Kinga Grabarczyk** (Inżynier Danych)")
st.sidebar.write("**Irek Kosek** (Specjalista HCI)")

st.sidebar.markdown("---")
st.sidebar.write("**Prowadzący:** dr inż. Adrian Kapczyński")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model_path = "models/best.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    except:  # noqa: E722
        st.error("Nie znaleziono pliku modelu .pth. Sprawdź ścieżkę!")
        return None, device


model, device = load_model()

uploaded_file = st.file_uploader("Wgraj skan CT płuc (PNG)", type=["png", "jpg"])

if uploaded_file is not None and model is not None:
    img = Image.open(uploaded_file).convert("L")
    img_array = np.array(img)

    process = Compose(
        [
            EnsureChannelFirst(channel_dim="no_channel"),
            ScaleIntensity(minv=0.0, maxv=1.0),
            Resize(spatial_size=(512, 512)),
        ]
    )

    processed = process(img_array)
    input_tensor = torch.from_numpy(np.array(processed)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor))
        prediction = (output > 0.5).float().squeeze().cpu().numpy()

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Oryginalny skan")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Wynik segmentacji")
        fig_pred, ax_pred = plt.subplots()
        ax_pred.imshow(img_array, cmap="gray")
        ax_pred.imshow(prediction, cmap="Reds", alpha=0.5)
        ax_pred.axis("off")
        st.pyplot(fig_pred)

    with col3:
        st.subheader("Mapa prawdopodobieństwa")
        prob_map = output.squeeze().cpu().numpy()
        fig_prob, ax_prob = plt.subplots()
        im = ax_prob.imshow(prob_map, cmap="hot")
        plt.colorbar(im, ax=ax_prob)
        ax_prob.axis("off")
        st.pyplot(fig_prob)

    st.success(
        "Analiza zakończona. Model zidentyfikował obszary podejrzane o zmiany nowotworowe."
    )
