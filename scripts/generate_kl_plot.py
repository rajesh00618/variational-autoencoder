"""
Script to calculate and visualize the KL divergence per latent dimension.

Usage:
    python scripts/generate_kl_plot.py
"""

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vae_model import VAE


@torch.no_grad()
def calculate_kl_per_dimension(
    model: VAE,
    test_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Calculate the mean KL divergence for each latent dimension
    across the entire test set.

    Uses the analytical formula:
    KLD_i = -0.5 * (1 + logvar_i - mu_i^2 - exp(logvar_i))
    averaged over all samples.
    """
    model.eval()
    kl_per_dim_sum = torch.zeros(model.latent_dim, device=device)
    num_samples = 0

    for data, _ in test_loader:
        data = data.to(device)
        mu, logvar = model.encoder(data)

        # KL per dimension: shape (batch_size, latent_dim)
        kl_per_sample = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_dim_sum += kl_per_sample.sum(dim=0)
        num_samples += data.size(0)

    kl_per_dim = kl_per_dim_sum.cpu().numpy() / num_samples
    return kl_per_dim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = VAE(latent_dim=8)
    model.load_state_dict(torch.load("models/vae.pt", map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded (latent_dim={model.latent_dim})")

    # Load test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    print(f"Test dataset size: {len(test_dataset)}")

    # Calculate KL per dimension
    kl_per_dim = calculate_kl_per_dimension(model, test_loader, device)
    print(f"\nKL divergence per latent dimension:")
    for i, kl in enumerate(kl_per_dim):
        status = "ACTIVE" if kl > 0.01 else "DEAD"
        print(f"  Dim {i}: {kl:.6f} [{status}]")

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    dims = np.arange(len(kl_per_dim))
    colors = ["#4CAF50" if kl > 0.01 else "#F44336" for kl in kl_per_dim]
    ax.bar(dims, kl_per_dim, color=colors, edgecolor="black", linewidth=1.2)
    ax.set_xlabel("Latent Dimension", fontsize=12)
    ax.set_ylabel("Mean KL Divergence", fontsize=12)
    ax.set_title("KL Divergence per Latent Dimension", fontsize=14, fontweight="bold")
    ax.set_xticks(dims)
    ax.axhline(y=0.01, color="orange", linestyle="--", alpha=0.7, label="Active threshold")
    ax.legend(fontsize=10)

    # Annotate bars
    for i, kl in enumerate(kl_per_dim):
        ax.text(i, kl + 0.005, f"{kl:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    output_path = "results/kl_per_dimension.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {output_path}")

    # Count active/dead dimensions
    active_dims = sum(1 for kl in kl_per_dim if kl > 0.01)
    dead_dims = model.latent_dim - active_dims
    print(f"\nActive dimensions: {active_dims}/{model.latent_dim}")
    print(f"Dead dimensions (posterior collapse): {dead_dims}/{model.latent_dim}")


if __name__ == "__main__":
    main()
