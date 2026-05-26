"""
Interactive Streamlit application for exploring a trained VAE's latent space.

Features:
- Latent space map (PCA/t-SNE 2D projection)
- Interactive latent dimension sliders for real-time image generation
- Reconstruction viewer with error heatmaps
- KL per dimension diagnostic analysis
"""

import os

import streamlit as st
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from vae_model import VAE

# Page config
st.set_page_config(
    page_title="VAE Latent Space Explorer",
    page_icon="🧬",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load the trained VAE model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=8)
    model.load_state_dict(torch.load("models/vae.pt", map_location=device))
    model.to(device)
    model.eval()
    return model, device


@st.cache_data
def load_test_data():
    """Load the test dataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    all_images, all_labels = next(iter(test_loader))
    return all_images, all_labels, test_dataset


@st.cache_data
def compute_latent_vectors(_model, _all_images, device):
    """Compute latent mean vectors for all test images."""
    _all_images = _all_images.to(device)
    with torch.no_grad():
        mu, _ = _model.encoder(_all_images)
    return mu.cpu().numpy()


@st.cache_data
def compute_pca(latent_vectors):
    """Compute 2D PCA projection of latent vectors."""
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)
    return latent_2d, pca


@st.cache_data
def compute_kl_per_dimension(_model, _test_loader, device):
    """Compute mean KL divergence for each latent dimension."""
    _model.eval()
    kl_sum = torch.zeros(_model.latent_dim, device=device)
    num_samples = 0

    with torch.no_grad():
        for data, _ in _test_loader:
            data = data.to(device)
            mu, logvar = _model.encoder(data)
            kl_per_sample = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_sum += kl_per_sample.sum(dim=0)
            num_samples += data.size(0)

    return (kl_sum / num_samples).cpu().numpy()


def get_class_names():
    """FashionMNIST class names."""
    return [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]


# Title
st.title("🧬 VAE Latent Space Explorer")
st.markdown(
    "Explore a Variational Autoencoder trained on FashionMNIST. "
    "Use the interactive tools below to understand the latent space."
)

# Load model and data
with st.spinner("Loading model and data..."):
    model, device = load_model()
    all_images, all_labels, test_dataset = load_test_data()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    latent_vectors = compute_latent_vectors(model, all_images, device)
    latent_2d, pca = compute_pca(latent_vectors)
    kl_per_dim = compute_kl_per_dimension(model, test_loader, device)

class_names = get_class_names()
label_names = [class_names[l] for l in all_labels.numpy()]


# ---- Sidebar ----
st.sidebar.header("🎛️ Latent Dimension Controls")
st.sidebar.markdown(
    "Adjust the sliders to explore each dimension of the latent space. "
    "The decoder will generate an image in real-time."
)

# Determine slider ranges from latent data
latent_ranges = []
for d in range(model.latent_dim):
    dim_data = latent_vectors[:, d]
    lo = float(np.percentile(dim_data, 5))
    hi = float(np.percentile(dim_data, 95))
    # Expand range slightly
    margin = (hi - lo) * 0.3
    latent_ranges.append((lo - margin, hi + margin))

# Sliders for each latent dimension
slider_values = []
for d in range(model.latent_dim):
    lo, hi = latent_ranges[d]
    val = st.sidebar.slider(
        f"z[{d}]", min_value=float(f"{lo:.2f}"), max_value=float(f"{hi:.2f}"),
        value=0.0, step=0.1, key=f"slider_{d}"
    )
    slider_values.append(val)

# Generate image from sliders
z = torch.tensor([slider_values], dtype=torch.float32, device=device)
with torch.no_grad():
    generated = model.decoder(z).cpu().squeeze().numpy()

# Display generated image
st.sidebar.header("🖼️ Generated Image")
st.sidebar.image(generated, caption="Generated from slider values", width=150)

# Reset button
if st.sidebar.button("Reset All Sliders"):
    for d in range(model.latent_dim):
        st.session_state[f"slider_{d}"] = 0.0
    st.rerun()


# ---- Main Content Tabs ----
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Latent Space Map",
    "🖼️ Reconstruction Viewer",
    "📊 KL per Dimension",
    "📈 Training Curves",
])

# ---- Tab 1: Latent Space Map ----
with tab1:
    st.header("Latent Space Map (PCA 2D Projection)")
    st.markdown(
        "Each point represents a test image projected into 2D using PCA. "
        "Color indicates the class label. Hover over points to see details."
    )

    df = pd.DataFrame({
        "PC1": latent_2d[:, 0],
        "PC2": latent_2d[:, 1],
        "Label": label_names,
        "Index": range(len(all_labels)),
    })

    fig = px.scatter(
        df, x="PC1", y="PC2", color="Label",
        hover_data=["Index"],
        title="2D Latent Space of FashionMNIST",
        color_discrete_sequence=px.colors.qualitative.Set2,
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        legend=dict(itemsizing="constant"),
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Click a point to decode")
    st.markdown(
        "Enter a test image index to see its original and reconstructed images below."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_idx = st.number_input(
            "Test image index",
            min_value=0,
            max_value=len(test_dataset) - 1,
            value=0,
            step=1,
        )

    with col2:
        st.markdown("---")

    if selected_idx is not None:
        orig_img, orig_label = test_dataset[selected_idx]
        orig_np = orig_img.squeeze().numpy()

        with torch.no_grad():
            img_tensor = orig_img.unsqueeze(0).to(device)
            recon, _, _ = model(img_tensor)
            recon_np = recon.cpu().squeeze().numpy()

        error = np.abs(orig_np - recon_np)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.image(orig_np, caption=f"Original ({class_names[orig_label]})", width=150)
        with col_b:
            st.image(recon_np, caption="Reconstructed", width=150)
        with col_c:
            fig_err, ax = plt.subplots(figsize=(2, 2))
            im = ax.imshow(error, cmap="hot", vmin=0, vmax=1)
            ax.set_title("Error Heatmap")
            ax.axis("off")
            st.pyplot(fig_err)
            plt.close()


# ---- Tab 2: Reconstruction Viewer ----
with tab2:
    st.header("Reconstruction Viewer")
    st.markdown(
        "Select random images from the test set and see how the VAE reconstructs them. "
        "The error heatmap highlights regions where the reconstruction differs most."
    )

    if st.button("Show Random Reconstruction"):
        idx = np.random.randint(len(test_dataset))
        orig_img, orig_label = test_dataset[idx]
        orig_np = orig_img.squeeze().numpy()

        with torch.no_grad():
            img_tensor = orig_img.unsqueeze(0).to(device)
            recon, _, _ = model(img_tensor)
            recon_np = recon.cpu().squeeze().numpy()

        error = np.abs(orig_np - recon_np)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(orig_np, caption=f"Original ({class_names[orig_label]})", width=200)
        with col2:
            st.image(recon_np, caption="Reconstructed", width=200)
        with col3:
            fig_err, ax = plt.subplots(figsize=(2.5, 2.5))
            im = ax.imshow(error, cmap="hot", vmin=0, vmax=1)
            ax.set_title("Error Heatmap")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
            st.pyplot(fig_err)
            plt.close()

        st.caption(f"Image index: {idx} | MSE: {np.mean(error**2):.6f}")

    # Side-by-side comparison for multiple indices
    st.subheader("Batch Comparison")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            idx = i * 100
            if idx < len(test_dataset):
                orig_img, orig_label = test_dataset[idx]
                orig_np = orig_img.squeeze().numpy()
                with torch.no_grad():
                    img_tensor = orig_img.unsqueeze(0).to(device)
                    recon, _, _ = model(img_tensor)
                    recon_np = recon.cpu().squeeze().numpy()

                st.image(orig_np, caption=f"Original [{idx}]", width=100)
                st.image(recon_np, caption=f"Recon [{idx}]", width=100)


# ---- Tab 3: KL per Dimension ----
with tab3:
    st.header("KL Divergence per Latent Dimension")
    st.markdown(
        "This bar chart shows the mean KL divergence for each latent dimension "
        "calculated over the entire test set. **Dimensions with near-zero KL are "
        "not being used by the model** — a clear sign of posterior collapse."
    )

    fig_kl, ax = plt.subplots(figsize=(10, 5))
    dims = np.arange(len(kl_per_dim))
    colors = ["#4CAF50" if kl > 0.01 else "#F44336" for kl in kl_per_dim]
    bars = ax.bar(dims, kl_per_dim, color=colors, edgecolor="black", linewidth=1.2)
    ax.set_xlabel("Latent Dimension", fontsize=12)
    ax.set_ylabel("Mean KL Divergence", fontsize=12)
    ax.set_title("KL Divergence per Latent Dimension", fontsize=14, fontweight="bold")
    ax.set_xticks(dims)
    ax.axhline(y=0.01, color="orange", linestyle="--", alpha=0.7, label="Active threshold")
    ax.legend(fontsize=10)

    for i, kl in enumerate(kl_per_dim):
        ax.text(i, kl + 0.005, f"{kl:.3f}", ha="center", va="bottom", fontsize=9)

    st.pyplot(fig_kl)
    plt.close()

    # Summary stats
    active_dims = sum(1 for kl in kl_per_dim if kl > 0.01)
    dead_dims = model.latent_dim - active_dims

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Dimensions", model.latent_dim)
    col2.metric("Active Dimensions", active_dims, delta_color="normal")
    col3.metric("Dead Dimensions (Collapse)", dead_dims, delta_color="inverse")


# ---- Tab 4: Training Curves ----
with tab4:
    st.header("Training Curves")
    st.markdown(
        "The training log shows both reconstruction loss and KL divergence over epochs. "
        "The KL curve should show a gradual increase due to annealing."
    )

    csv_path = "results/training_log.csv"
    if os.path.exists(csv_path):
        df_log = pd.read_csv(csv_path)

        fig_curves, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Reconstruction loss
        ax1.plot(df_log["epoch"], df_log["reconstruction_loss"], "b-", linewidth=2)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Reconstruction Loss", fontsize=12)
        ax1.set_title("Reconstruction Loss", fontsize=14, fontweight="bold")
        ax1.grid(alpha=0.3)

        # KL divergence
        ax2.plot(df_log["epoch"], df_log["kl_divergence"], "r-", linewidth=2)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("KL Divergence", fontsize=12)
        ax2.set_title("KL Divergence", fontsize=14, fontweight="bold")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig_curves)
        plt.close()

        with st.expander("View Raw Training Log"):
            st.dataframe(df_log)
    else:
        st.warning("Training log not found at results/training_log.csv")
