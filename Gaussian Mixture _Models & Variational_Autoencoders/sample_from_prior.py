
import numpy as np
import matplotlib.pyplot as plt
import torch

@torch.no_grad()
def sample_from_prior(vae, device, n=16, outpath="results/Prior_Samples.png"):
    vae.eval()
    latent_dim = vae.encoder.fc_mu.out_features
    z = torch.randn(n, latent_dim, device=device)
    recon = vae.decoder(z).squeeze(1).cpu().numpy()
    nrow = int(np.sqrt(n))
    fig, axs = plt.subplots(nrow, nrow, figsize=(nrow, nrow))
    idx = 0
    for i in range(nrow):
        for j in range(nrow):
            axs[i, j].imshow(recon[idx], cmap="gray"); axs[i, j].axis("off"); idx += 1
    plt.suptitle("Samples from prior (decoded)")
    plt.savefig(outpath, bbox_inches="tight", dpi=150); plt.close()
