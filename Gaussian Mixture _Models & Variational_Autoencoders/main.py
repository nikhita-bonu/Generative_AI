
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random, numpy as np, torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

from VAE import VAE
from train import train_epoch
from latent_traversal import latent_traversal
from sample_from_prior import sample_from_prior
from tsne import tsne_latent_mnist
from GMM_entropy import run_gmm_experiments

GLOBAL_SEED = 1337
SKIP_EPOCH0_PLOTS = True
SAVE_EVERY_N_EPOCHS = 5
latent_dims_to_run = [1, 10]
epochs_per_run = 10

def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def save_ckpt(vae, tag: str, epoch: int):
    Path("results/models").mkdir(exist_ok=True, parents=True)
    torch.save(vae.state_dict(), f"results/models/vae_{tag}_epoch{epoch}.pt")

def run_training(latent_dim=2, epochs=5, device=None, tag="dim2"):
    Path("results").mkdir(exist_ok=True, parents=True)
    vae = VAE(latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    train_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=128, shuffle=True
    )
    elbo_curve = []

    if not SKIP_EPOCH0_PLOTS:
        latent_traversal(vae, device, outpath=f"results/Latent_Traversal_{tag}_epoch0.png")
    sample_from_prior(vae, device, outpath=f"results/Prior_Samples_{tag}_epoch0.png")
    tsne_latent_mnist(vae, device, outpath=f"results/Latent_TSNE_{tag}_epoch0.png")

    mid_epoch = max(1, epochs // 2)

    for epoch in range(1, epochs+1):
        loss = train_epoch(vae, train_loader, optimizer, device)
        elbo_curve.append(loss)
        print(f"[{tag}] Epoch {epoch}/{epochs} | ELBO Loss: {loss:.4f}")

        if SAVE_EVERY_N_EPOCHS and epoch % SAVE_EVERY_N_EPOCHS == 0:
            save_ckpt(vae, tag, epoch)

        if epoch == mid_epoch:
            latent_traversal(vae, device, outpath=f"results/Latent_Traversal_{tag}_epoch{epoch}.png")
    sample_from_prior(vae, device, outpath=f"results/Prior_Samples_{tag}_epoch{epoch}.png")
    tsne_latent_mnist(vae, device, outpath=f"results/Latent_TSNE_{tag}_epoch{epoch}.png")

    latent_traversal(vae, device, outpath=f"results/Latent_Traversal_{tag}_epoch{epochs}.png")
    sample_from_prior(vae, device, outpath=f"results/Prior_Samples_{tag}_epoch{epochs}.png")
    tsne_latent_mnist(vae, device, outpath=f"results/Latent_TSNE_{tag}_epoch{epochs}.png")
    torch.save(vae.state_dict(), f"results/models/vae_{tag}_final.pt")

    plt.figure(); plt.plot(range(1, epochs+1), elbo_curve, marker='o')
    plt.title(f"ELBO Learning Curve ({tag})"); plt.xlabel("Epoch"); plt.ylabel("ELBO Loss"); plt.grid(alpha=0.3)
    plt.savefig(f"results/ELBO_Curve_{tag}.png", bbox_inches="tight", dpi=150); plt.close()

def main():
    Path("results").mkdir(exist_ok=True, parents=True)
    set_all_seeds(GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\\n[INFO] Running GMM experiments...")
    run_gmm_experiments(outdir="results", seed=GLOBAL_SEED)

    print("\\n[INFO] Training VAE...")
    for d in latent_dims_to_run:
        run_training(latent_dim=d, epochs=epochs_per_run, device=device, tag=f"dim{d}")
    print("\\nDone. Check results/ and results/models/")

if __name__ == "__main__":
    main()
