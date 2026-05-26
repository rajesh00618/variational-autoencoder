"""
Script to generate a sample reconstruction and its corresponding error heatmap.

Usage:
    python scripts/generate_reconstruction.py --index 10
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vae_model import VAE


def main():
    parser = argparse.ArgumentParser(
        description="Generate reconstruction and error heatmap for a test image"
    )
    parser.add_argument("--index", type=int, default=10, help="Test image index")
    parser.add_argument(
        "--model-path", type=str, default="models/vae.pt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save output images"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = VAE(latent_dim=8)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Get the specified test image
    if args.index >= len(test_dataset):
        print(f"Error: index {args.index} exceeds test dataset size ({len(test_dataset)})")
        return

    original_img, label = test_dataset[args.index]
    original_img = original_img.unsqueeze(0).to(device)  # Add batch dim

    # Reconstruct
    with torch.no_grad():
        recon_img, mu, logvar = model(original_img)

    # Convert to numpy for visualization
    original_np = original_img.cpu().squeeze().numpy()
    recon_np = recon_img.cpu().squeeze().numpy()

    # Compute per-pixel error (absolute difference)
    error = np.abs(original_np - recon_np)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save original image
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(original_np, cmap="gray")
    ax.set_title(f"Original (label={label})")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"original_{args.index}.png"), dpi=150)
    plt.close()

    # Save reconstructed image
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(recon_np, cmap="gray")
    ax.set_title("Reconstructed")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"reconstructed_{args.index}.png"), dpi=150)
    plt.close()

    # Save error heatmap
    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(error, cmap="hot", vmin=0, vmax=1)
    ax.set_title("Reconstruction Error")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"heatmap_{args.index}.png"), dpi=150)
    plt.close()

    print(f"Saved images to {args.output_dir}/")
    print(f"  - original_{args.index}.png")
    print(f"  - reconstructed_{args.index}.png")
    print(f"  - heatmap_{args.index}.png")
    print(f"\nReconstruction error (MSE): {np.mean(error**2):.6f}")


if __name__ == "__main__":
    main()
