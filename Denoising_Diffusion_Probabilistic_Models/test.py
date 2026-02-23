import torch
import matplotlib.pyplot as plt
import numpy as np

from train import FullyConnectedNet, DiffusionSampler, sample_target

if __name__ == '__main__':
    device = 'cpu'

    beta_schedule = "cosine" 
    h_size = 128
    timesteps = 500
    model =  FullyConnectedNet(3, [h_size] * 3, 2, timesteps, 'relu')
    sampler = DiffusionSampler(timesteps, 'cosine')

    ## load in the model
    ## Task1.3 change
    model_path = f"results/model_{beta_schedule}_{timesteps}.pt"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    model.eval()

    # Grid of points in 2D
    x_range = np.linspace(-7, 7, 20)
    y_range = np.linspace(-7, 7, 20)
    xx, yy = np.meshgrid(x_range, y_range)
    grid = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)

    ## sample data of a particular shape
    x0 = sample_target(100)
    ## create random samples from a normal distribution (same shape as x0)
    xT = torch.randn_like(x0)

    num_images = 2
    ##x0_hat, times = sampler.p_sample_loop(model, xT, num_images =  num_images)
    sequence, times = sampler.p_sample_loop(model, xT, num_images=num_images)
    ## create the denoising plot.
    fig, ax = plt.subplots(1, len(sequence), figsize = (num_images * 3, 2))

    for i in range(len(sequence)):
        ax[i].set_title(f't = {times[i]}')
        ax[i].set_xlim(-7.0, 7.0)
        ax[i].set_ylim(-7.0, 7.0)

        ## vector field
        t = torch.full((grid_tensor.shape[0],), times[i], dtype=torch.long, device=device)

        print(t)

        # Evaluate the model to get the score function
        with torch.no_grad():
            score = - model(grid_tensor, t)

        # Reshape the vector field for plotting
        u = score[:, 0].cpu().numpy().reshape(xx.shape)
        v = score[:, 1].cpu().numpy().reshape(yy.shape)
        ax[i].quiver(
            xx, yy, u, v,
            angles='xy',
            scale_units='xy',
            scale=20,                 # or try a smaller scale like 10 to lengthen arrows
            width=0.005,              # makes the shaft thicker
            headwidth=4,              # wider arrow head
            headlength=6,             # longer arrow head
            headaxislength=5,         # axis length of the head
            color='k',                # black, or pick any strong color
            alpha=1.0,                # fully opaque
            linewidth=1.0             # you can bump this up to 1.5 or 2.0
        )

        if i == 0:
            ax[0].scatter(xT[:, 0], xT[:, 1], alpha = 0.5, s = 10)
        else:
            ax[i].scatter(sequence[i][:, 0], sequence[i][:, 1], alpha = 0.5, s = 10)
        ax[i].axis('off')


    fig.tight_layout()
    fig.savefig('denoise_moons.png', dpi = 500)
    plt.show()
