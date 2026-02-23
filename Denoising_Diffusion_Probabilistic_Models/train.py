'''
Homework: Complete the diffusion model
---------------------------------------

In this assignment, you will implement and train a diffusion model on a simple dataset.

Tasks:
1. Implement the forward diffusion process (q_sample)
2. Implement the reverse denoising process (p_sample)
3. Implement the DDPM training loss
4. Train the model and visualize the learning curve
5. Run the test file to visualize the model's noise prediction

References:
- Ho et al., "Denoising Diffusion Probabilistic Models" (https://arxiv.org/pdf/2006.11239)
'''

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from sklearn import datasets
import matplotlib.pyplot as plt

def add_singletons_like(v: Tensor, x_shape: tuple) -> Tensor:
    ## get batchsize (b) and data dimention (d ...)
    b, *d = x_shape
    ## add singletons for each dim in data dim
    return v.reshape(b, *[1] * len(d))

def extract(v: Tensor, t: Tensor, x_shape: tuple) -> Tensor:
    return add_singletons_like(v.gather(0, t), x_shape)

def linear_beta_schedule(timesteps: int) -> Tensor:
    return torch.linspace(1e-4, 0.02, timesteps)

def scaled_linear_beta_schedule(timesteps: int) -> Tensor:
    '''
    linear schedule, proposed in original ddpm paper
    '''
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s = 0.008):
    '''
    https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    '''
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DiffusionSampler(nn.Module):
    def __init__(self, timesteps: int, beta_schedule: str = 'scaled_linear'):
        super().__init__()

        ## computing scheduling parameters
        if beta_schedule == 'linear':
            beta = linear_beta_schedule(timesteps)
        elif beta_schedule == 'scaled_linear':
            beta = scaled_linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            beta = cosine_beta_schedule(timesteps)
        else:
            raise ValueError('Unrecognized beta_schedule.')
        alpha = 1.0 - beta
        alpha_bar = alpha.cumprod(dim = 0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value = 1.)

        ## adding as non trainable parameters
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('alpha_bar_prev', alpha_bar_prev)

    @property
    def timesteps(self):
        return len(self.beta)


    @property
    def device(self):
        return self.beta.device
    
    @property
    def timesteps(self):
        return len(self.beta)
    
    @torch.no_grad()
    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor = None):
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar = extract(self.alpha_bar, t, x0.shape)

        ## compute the forward sample q(x_t | x_0)
        ## draw a sample x_t ~ q(x_t | x_0)
        '''
        To-Do:
        See Equation (4) in: https://arxiv.org/pdf/2006.11239

        Also see Algoithm 1 (Training)
        '''
        ## Task1.1 change
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise

        
        return xt, noise
    
    @torch.no_grad()
    def p_sample(self, model: nn.Module, xt: Tensor, t: Tensor) -> Tensor:
        beta               = extract(self.beta, t, xt.shape)
        alpha              = extract(self.alpha, t, xt.shape)
        alpha_bar          = extract(self.alpha_bar, t, xt.shape)
        alpha_bar_prev     = extract(self.alpha_bar_prev, t, xt.shape)
        not_first_timestep = add_singletons_like(t > 0, xt.shape)
        ## compute mu and variance of the reverse sampling distribution p(x_{t-1} | x_t)
        ## draw a sample x_{t-1} ~ p(x_{t-1} | x_t)
        '''
        To-Do:
        See Equation (7) in: https://arxiv.org/pdf/2006.11239

        Also see Algorithm (2) Sampling
        '''

        ## Task 1.1 change
        eps_theta = model(xt, t)
        sqrt_alpha = torch.sqrt(alpha)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

        mu_theta = (1.0 / sqrt_alpha) * (
            xt - (beta / sqrt_one_minus_alpha_bar) * eps_theta
        )
        beta_tilde = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * beta
        noise = torch.randn_like(xt)

        sample = mu_theta + not_first_timestep * torch.sqrt(beta_tilde) * noise
        return sample
    
    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, xT: Tensor, num_images: int = 1) -> Tensor:

        xt = xT.clone()

        if num_images > 1:
            dt = int(self.timesteps / num_images)
            sequence = []
            times = []

        for i in reversed(range(self.timesteps)):
            t = torch.ones(xT.shape[0], device = xT.device).long() * i
            xt = self.p_sample(model, xt, t)

            if num_images > 1 and (i % dt == 0 or i == self.timesteps - 1):
                sequence.append(xt)
                times.append(i)

        if num_images > 1:
            return sequence, times
        else:
            return xt
        
class FullyConnectedNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layer_dims: list,
        output_dim: int,
        timesteps: int,
        activation: callable = None
    ):
        super().__init__()

        self.timesteps = timesteps
        self.layers = nn.Sequential()
        in_dim = input_dim
        if not activation:
            activation = 'RELU'
        for out_dim in hidden_layer_dims:
            self.layers.append(nn.Linear(in_dim, out_dim))
            if activation.upper() == 'RELU':
                self.layers.append(nn.ReLU())
            elif activation.upper() == 'TANH':
                self.layers.append(nn.Tanh())
            else:
                raise Exception("Invalid activation function")
            in_dim = out_dim

        self.layers.append(nn.Linear(in_dim, output_dim))

    def forward(self, x_t, time):
        time = time.reshape((len(time), 1)) / self.timesteps
        x = torch.cat((time, x_t), 1)
        x = self.layers(x)
        return x
        
def sample_target(n_samples: int, device = 'cpu'):
    x0 = datasets.make_moons(n_samples = n_samples, noise = 0.05)[0] * 2.5
    x0 = torch.tensor(x0, dtype = torch.float32, device = device)
    return x0

if __name__ == '__main__':
    device = 'cpu'

    h_size = 128
    timesteps = 500
    batch_size = 1024
    steps = 5000

    model =  FullyConnectedNet(3, [h_size] * 3, 2, timesteps, 'relu').to(device)
    sampler = DiffusionSampler(timesteps, 'scaled_linear')
    opt = torch.optim.Adam(model.parameters())

    ## main training loop
    loss_hist = []
    for i in range(steps):
        opt.zero_grad()
        t = torch.randint(0, timesteps, (batch_size,), dtype = torch.long, device = device)
        x0 = sample_target(batch_size)
        ## compute loss
        '''
        To-Do:
        See Algorithm 1 (Training): https://arxiv.org/pdf/2006.11239
        '''

        ## Task 1.1 change
        
        xt, noise = sampler.q_sample(x0, t)
        noise_pred = model(xt, t)

        loss = F.mse_loss(noise_pred, noise)
        



        loss.backward()
        opt.step()
        print(i, loss.item())
        loss_hist.append(loss.item())

    ## save the trained model weights
    torch.save(model.state_dict(), 'model.pt')

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Train Step')
    ax.set_ylabel('Diffusion Loss')
    ax.plot(loss_hist)
    fig.tight_layout()
    fig.savefig('loss.png', dpi = 300)
    plt.show()
