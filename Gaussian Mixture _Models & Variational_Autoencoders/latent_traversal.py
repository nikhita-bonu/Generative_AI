
import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def latent_traversal(vae, device, z_min=-3, z_max=3, steps=15, outpath="results/Latent_Traversal.png"):
    vae.eval()
    latent_dim = vae.encoder.fc_mu.out_features
    z = torch.zeros(steps, latent_dim, device=device)
    z[:, 0] = torch.linspace(z_min, z_max, steps, device=device)
    recon = vae.decoder(z).squeeze(1).cpu().numpy()
    fig, axs = plt.subplots(1, steps, figsize=(steps, 2))
    for i in range(steps):
        axs[i].imshow(recon[i], cmap="gray"); axs[i].axis("off")
    plt.suptitle("Latent traversal (varying z[0])")
    plt.savefig(outpath, bbox_inches="tight", dpi=150); plt.close()
