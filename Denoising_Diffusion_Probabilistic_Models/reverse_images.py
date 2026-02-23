import torch
import matplotlib.pyplot as plt
from unet import UNet2D
from train import DiffusionSampler

timesteps = 200         
beta_schedule = "cosine"    
model_path = f"results/mnist_unet_{beta_schedule}_{timesteps}.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet2D(
    in_ch=1,
    base=64,                
    time_embed_dim=128,
    timesteps=timesteps
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
sampler = DiffusionSampler(timesteps, beta_schedule).to(device)
xT = torch.randn((1, 1, 28, 28), device=device)

sequence, times = sampler.p_sample_loop(model, xT, num_images=6)

fig, ax = plt.subplots(1, len(sequence), figsize=(15, 3))
for i, xt in enumerate(sequence):
    img = xt[0].detach().cpu().squeeze()
    ax[i].imshow((img + 1) / 2, cmap="gray")
    ax[i].axis("off")
    ax[i].set_title(f"t={times[i]}")

fig.tight_layout()
save_path = f"results/reverse_{beta_schedule}_{timesteps}.png"
plt.savefig(save_path, dpi=300)
plt.close()

print("Saved:", save_path)
