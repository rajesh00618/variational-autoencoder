"""
Training script for the Variational Autoencoder on FashionMNIST.

Implements KL annealing to prevent posterior collapse and logs
reconstruction loss and KL divergence separately for diagnosis.
"""

import os
import csv
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vae_model import VAE, vae_loss


def get_data_loaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """Load FashionMNIST train and test DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_epoch(
    model: VAE,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    beta: float,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Train the VAE for one epoch.

    Returns:
        Tuple of (avg_total_loss, avg_recon_loss, avg_kl_div).
    """
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_samples = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss, recon_loss, kl_div = vae_loss(recon_batch, data, mu, logvar, beta)

        loss.backward()
        optimizer.step()

        batch_size = data.size(0)
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_div.item()
        num_samples += batch_size

    return (
        total_loss / num_samples,
        total_recon / num_samples,
        total_kl / num_samples,
    )


@torch.no_grad()
def evaluate(
    model: VAE,
    test_loader: DataLoader,
    beta: float,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate the VAE on the test set."""
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_samples = 0

    for data, _ in test_loader:
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
        loss, recon_loss, kl_div = vae_loss(recon_batch, data, mu, logvar, beta)

        batch_size = data.size(0)
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_div.item()
        num_samples += batch_size

    return (
        total_loss / num_samples,
        total_recon / num_samples,
        total_kl / num_samples,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a VAE on FashionMNIST")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--latent-dim", type=int, default=8, help="Latent space dimensionality")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-annealing-epochs", type=int, default=20,
                        help="Number of epochs for KL annealing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = get_data_loaders(args.batch_size)

    # Model
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training log
    os.makedirs("results", exist_ok=True)
    csv_path = "results/training_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "reconstruction_loss", "kl_divergence"])

    print(f"Training VAE with latent_dim={args.latent_dim}, epochs={args.epochs}")
    print(f"KL annealing over {args.num_annealing_epochs} epochs")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # KL annealing: linear schedule
        beta = min(1.0, epoch / args.num_annealing_epochs)

        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, beta, device
        )
        test_loss, test_recon, test_kl = evaluate(
            model, test_loader, beta, device
        )

        # Log to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, round(test_recon, 4), round(test_kl, 4)])

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | beta={beta:.3f} | "
                f"Train Recon: {train_recon:.4f} | Train KL: {train_kl:.4f} | "
                f"Test Recon: {test_recon:.4f} | Test KL: {test_kl:.4f}"
            )

    # Save model checkpoint
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/vae.pt")
    print(f"\nModel saved to models/vae.pt")
    print(f"Training log saved to {csv_path}")


if __name__ == "__main__":
    main()
