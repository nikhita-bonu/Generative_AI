
import os
import torch
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from train import DiffusionSampler
from unet import UNet2D


if __name__ == "__main__":
    
    os.makedirs("results", exist_ok=True)
    timesteps = 200                  
    beta_schedule = "cosine"         
    batch_size = 128
    epochs = 20                      
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = UNet2D(
        in_ch=1,
        base=64,                     
        time_embed_dim=128,
        timesteps=timesteps
    ).to(device)

    sampler = DiffusionSampler(timesteps, beta_schedule).to(device)
    opt = optim.Adam(model.parameters(), lr=learning_rate)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))       
    ])

    train_ds = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    print("Starting training...")
    loss_hist = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for x, _ in train_dl:
            x = x.to(device)
            b = x.shape[0]

            t = torch.randint(
                low=0,
                high=timesteps,
                size=(b,),
                device=device
            )

            # Forward diffusion
            xt, noise = sampler.q_sample(x, t)
            noise_pred = model(xt, t)
            loss = F.mse_loss(noise_pred, noise)

            opt.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_hist.append(loss.item())

        print(f"  Loss: {loss_hist[-1]:.4f}")

    model_path = f"mnist_unet_{beta_schedule}_{timesteps}.pt"
    torch.save(model.state_dict(), os.path.join("results", model_path))
    print("Saved model to:", model_path)

    plt.figure(figsize=(6, 4))
    plt.plot(loss_hist)
    plt.title("MNIST DDPM Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f"results/mnist_loss_{beta_schedule}_{timesteps}.png", dpi=300)
    plt.close()

    print("Saved loss curve.")
