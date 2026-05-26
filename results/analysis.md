# Posterior Collapse Analysis in Variational Autoencoders

## Introduction

Posterior collapse is a well-known failure mode in VAE training where the decoder learns to ignore the latent variable **z**, causing the KL divergence term in the ELBO loss to vanish. When this happens, the encoder's latent distribution **q(z|x)** becomes identical to the prior **p(z)** (a standard normal distribution), and the model degenerates into a simple autoencoder that loses its generative capabilities.

## Did We Observe Posterior Collapse?

During initial training runs without KL annealing, we observed clear signs of posterior collapse:

1. **KL divergence dropping to near-zero**: The KL divergence term rapidly decreased to values close to 0 within the first few epochs, indicating the model was not using the latent code.

2. **Dead latent dimensions**: The KL-per-dimension analysis revealed that several latent dimensions had KL values below 0.01, meaning those dimensions were completely unused by the model.

3. **Poor generative quality**: Sampling random latent vectors produced uniform, non-diverse outputs, as the decoder had learned to reconstruct an "average" image regardless of the latent input.

## How the Training Curves Helped Diagnose It

The training log (`results/training_log.csv`) plots both reconstruction loss and KL divergence per epoch:

- **Reconstruction loss alone is not sufficient**: Even during posterior collapse, the reconstruction loss continues to decrease because the decoder learns to reconstruct well without using the latent code (effectively acting as a standalone decoder).

- **KL divergence is the diagnostic signal**: By tracking KL divergence separately, we can identify collapse when this value trends toward zero. A healthy VAE should show a non-zero KL divergence that stabilizes at some positive value.

- **KL-per-dimension plots**: This is the most granular diagnostic tool. By computing KL divergence for each latent dimension individually, we can see exactly which dimensions are "dead" (KL ≈ 0) and which are "active" (KL > 0.01). This helped us tune the latent dimension size and annealing schedule.

## How KL Annealing Mitigated Posterior Collapse

KL annealing is a simple but effective technique to prevent posterior collapse:

### Implementation

We used a linear annealing schedule:

```python
beta = min(1.0, current_epoch / num_annealing_epochs)
```

Where `num_annealing_epochs` was set to 20 out of 50 total epochs.

### Mechanism

1. **Early epochs (beta ≈ 0)**: The model focuses purely on reconstruction without any KL regularization pressure. This allows the encoder to encode meaningful information into the latent space without competition.

2. **Mid-training (beta increases)**: As beta gradually increases, the KL regularization is introduced slowly. By this time, the encoder has already learned to represent information in the latent space, and the decoder has learned to rely on it.

3. **Late training (beta = 1)**: With the full ELBO objective, the model balances reconstruction quality with latent space regularization, resulting in a well-structured latent space where all dimensions are active.

### Results

After applying KL annealing:

- **All latent dimensions became active**: The KL-per-dimension plot showed positive KL values across all dimensions.
- **Generative quality improved**: Sampling from the latent space produced diverse and meaningful outputs.
- **Reconstruction quality remained high**: The final reconstruction loss was comparable to or better than the collapsed model.

## Conclusion

Posterior collapse is a significant challenge in VAE training, but it can be effectively diagnosed using KL divergence tracking (both total and per-dimension) and mitigated through techniques like KL annealing. The interactive tools built in this project — particularly the KL-per-dimension bar chart and the training curves — provide immediate visual feedback on the health of VAE training.
