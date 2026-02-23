
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

@torch.no_grad()
def tsne_latent_mnist(vae, device, n_samples=2000, outpath="results/Latent_TSNE_MNIST.png"):
    vae.eval()
    mnist = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    loader = DataLoader(mnist, batch_size=n_samples, shuffle=True)
    xs, ys = next(iter(loader))
    xs = xs.to(device)
    mu, _ = vae.encoder(xs)
    z = mu.cpu().numpy(); ys = ys.numpy()
    z2 = TSNE(n_components=2, init="random", learning_rate="auto").fit_transform(z)
    plt.figure(figsize=(8,8))
    sc = plt.scatter(z2[:,0], z2[:,1], c=ys, cmap="tab10", s=10, alpha=0.8)
    plt.legend(*sc.legend_elements(), title="Digits", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("t-SNE of VAE Latent Space (MNIST)"); plt.axis("off"); plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=150); plt.close()
