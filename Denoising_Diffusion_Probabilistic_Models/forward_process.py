import torch
import torchvision
import matplotlib.pyplot as plt

from train import DiffusionSampler

if __name__ == '__main__':

    x0 = torchvision.io.read_image('dog.jpeg').permute(1, 2, 0).rot90(k = 3).float().unsqueeze(0) / 255.0

    timesteps = 500
    cosine_sampler = DiffusionSampler(timesteps, 'cosine')
    linear_sampler = DiffusionSampler(timesteps, 'scaled_linear')

    num_steps = 5
    steps = torch.linspace(0, timesteps-1, num_steps, dtype = torch.long)

    fig, ax = plt.subplots(2, num_steps, figsize = (num_steps, 3))

    for i in range(num_steps):
        ax[0, i].axis('off')
        ax[1, i].axis('off')
        ax[0, i].set_title(f'Step: {steps[i].item()}', fontsize = 10)

        xt_cosine, _ = cosine_sampler.q_sample(x0, steps[i])
        xt_linear, _ = linear_sampler.q_sample(x0, steps[i])
        ax[0, i].imshow(xt_cosine[0])
        ax[1, i].imshow(xt_linear[0])
        print(i)

    fig.tight_layout()
    fig.savefig('annabelle.png', dpi = 300)
