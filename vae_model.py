"""
Variational Autoencoder (VAE) implementation in PyTorch.

This module defines the Encoder, Decoder, and VAE classes with the
reparameterization trick for differentiable sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Convolutional encoder that maps an input image to a latent distribution
    parameterized by mean (mu) and log-variance (logvar).
    """

    def __init__(self, latent_dim: int = 8):
        super().__init__()
        # Input: (batch, 1, 28, 28)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 4, 4)
            nn.ReLU(),
        )
        # After conv layers, feature map is (128, 4, 4) -> 128 * 4 * 4 = 2048
        conv_output_dim = 128 * 4 * 4
        self.fc_mu = nn.Linear(conv_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(conv_output_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """
    Convolutional decoder that reconstructs an image from a latent vector.
    Uses transposed convolutions to upsample back to the original image size.
    """

    def __init__(self, latent_dim: int = 8):
        super().__init__()
        # Project latent vector to a feature map suitable for transposed conv
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (64, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # (1, 32, 32)
            # Crop or interpolate to (1, 28, 28)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 128, 4, 4)
        x = self.deconv_layers(x)
        # Interpolate to exactly 28x28 (the deconv gives 32x32)
        x = F.interpolate(x, size=(28, 28), mode="bilinear", align_corners=False)
        x = torch.sigmoid(x)
        return x


class VAE(nn.Module):
    """
    Variational Autoencoder combining Encoder and Decoder with the
    reparameterization trick.
    """

    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon, where
        epsilon ~ N(0, I) and sigma = exp(0.5 * logvar).

        This allows gradients to flow through the sampling process.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    @torch.no_grad()
    def encode_to_mu(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent mean vectors (no gradients)."""
        mu, _ = self.encoder(x)
        return mu

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to reconstructed images (no gradients)."""
        return self.decoder(z)


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the ELBO (Evidence Lower Bound) loss for a VAE.

    Args:
        recon_x: Reconstructed images from the decoder.
        x: Original input images.
        mu: Mean of the latent distribution.
        logvar: Log-variance of the latent distribution.
        beta: Weight for the KL divergence term (for KL annealing).

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence).
    """
    # Reconstruction loss: Binary Cross-Entropy
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")

    # KL divergence: analytical formula
    # KLD = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_div
    return total_loss, recon_loss, kl_div
