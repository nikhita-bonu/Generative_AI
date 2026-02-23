
import torch

def _negative_log_likelihood(x, recon_mu):
    x = x.view(x.size(0), -1)
    recon_mu = recon_mu.view(recon_mu.size(0), -1)
    nll = 0.5 * torch.sum((x - recon_mu) ** 2, dim=1)
    return nll.mean()

def _kl_diag_normal(mu, logvar):
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()

def ELBO(x, recon_mu, mu, logvar):
    return _negative_log_likelihood(x, recon_mu) + _kl_diag_normal(mu, logvar)
