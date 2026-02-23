import torch
import matplotlib.pyplot as plt
from train import DiffusionSampler
from torchvision import datasets, transforms

timesteps = 200                 
beta_schedule = "cosine"        

device = "cuda" if torch.cuda.is_available() else "cpu"
sampler = DiffusionSampler(timesteps, beta_schedule).to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
x0, _ = dataset[0]          
x0 = x0.unsqueeze(0).to(device) 

steps_to_show = [0, timesteps//5, 2*timesteps//5,
                 3*timesteps//5, 4*timesteps//5, timesteps-1]

images = []

for t in steps_to_show:
    t_tensor = torch.tensor([t], device=device)
    xt, _ = sampler.q_sample(x0, t_tensor)
    images.append(xt[0].detach().cpu().squeeze())

# Plot results
fig, ax = plt.subplots(1, len(images), figsize=(15, 3))
for i, img in enumerate(images):
    ax[i].imshow((img + 1) / 2, cmap="gray")
    ax[i].axis("off")
    ax[i].set_title(f"t={steps_to_show[i]}")

fig.tight_layout()
save_path = f"results/forward_{beta_schedule}_{timesteps}.png"
plt.savefig(save_path, dpi=300)
plt.close()

print("Saved:", save_path)
