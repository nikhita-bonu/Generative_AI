
from ELBO import ELBO
def train_epoch(vae, loader, optimizer, device):
    vae.train()
    total = 0.0
    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon_mu, mu, logvar, _ = vae(x)
        loss = ELBO(x, recon_mu, mu, logvar)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)
