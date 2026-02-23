import torch
import os
from train import train_gan
from plotting import plot_losses, make_video, show_gan_training_grid

torch.manual_seed(41)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run(num_steps):
    if os.path.exists("frames"):
        files = sorted([f for f in os.listdir("frames") if f.endswith(".png")])
        for filename in files:
            os.remove(os.path.join("frames", filename))
    plot_every = 5
    losses = train_gan(device, num_steps=num_steps, plot_every=plot_every, mode="mixture4", batch_size=256)
    plot_losses(losses, plot_every=plot_every)
    # make_video(fps=20)

run(2500)
show_gan_training_grid(folder="frames", grid_rows=2, grid_cols=3, figsize=(14, 8), max_step=2500)
