
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from pathlib import Path
import json

def run_gmm_experiments(outdir="results", seed=42):
    np.random.seed(seed)
    Path(outdir).mkdir(exist_ok=True, parents=True)

    weights = np.array([0.2, 0.5, 0.3])
    means = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    covs = np.array([
        [[1.0, 0.3], [0.3, 0.8]],
        [[1.2, 0.0], [0.0, 1.2]],
        [[0.5,-0.2], [-0.2, 1.0]]
    ])

    N = 100_000
    comps = np.random.choice(3, size=N, p=weights)
    samples = np.zeros((N, 2), dtype=float)
    for k in range(3):
        idx = (comps == k)
        if np.any(idx):
            samples[idx] = np.random.multivariate_normal(mean=means[k], cov=covs[k], size=idx.sum())

    colors = ["r","g","b"]
    plt.figure(figsize=(8,8))
    for k, c in enumerate(colors):
        pts = samples[comps==k]
        plt.scatter(pts[:,0], pts[:,1], s=1, c=c, label=f"N(mu{k+1}, Sigma{k+1})")
    plt.legend(); plt.xlabel("x1"); plt.ylabel("x2")
    plt.title("GMM: 3-Component 2D Samples (N=100,000)")
    plt.savefig(f"{outdir}/GMM_scatter.png", bbox_inches="tight", dpi=150)
    plt.close()

    def log_mixture_pdf(x):
        logs = []
        for k in range(3):
            logs.append(np.log(weights[k]) + multivariate_normal.logpdf(x, mean=means[k], cov=covs[k]))
        m = np.max(logs)
        return m + np.log(np.sum(np.exp(np.array(logs) - m)))

    def estimate_entropy(sample_subset):
        logs = np.array([log_mixture_pdf(x) for x in sample_subset])
        return float(-np.mean(logs))

    sample_sizes = [1_000, 5_000, 10_000, 50_000, 100_000]
    entropies = [estimate_entropy(samples[:n]) for n in sample_sizes]

    plt.figure()
    plt.plot(sample_sizes, entropies, marker="o")
    plt.xscale("log")
    plt.xlabel("Number of samples (log scale)")
    plt.ylabel("Estimated entropy H(p)")
    plt.title("Entropy of GMM vs Number of Samples")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{outdir}/GMM_entropy.png", bbox_inches="tight", dpi=150)
    plt.close()

    deg_weights = np.array([1e-5, 1 - 2e-5, 1e-5])
    def log_mixture_pdf_deg(x):
        logs = []
        for k in range(3):
            logs.append(np.log(deg_weights[k]) + multivariate_normal.logpdf(x, mean=means[k], cov=covs[k]))
        m = np.max(logs)
        return m + np.log(np.sum(np.exp(np.array(logs) - m)))

    def estimate_entropy_deg(sample_subset):
        logs = np.array([log_mixture_pdf_deg(x) for x in sample_subset])
        return float(-np.mean(logs))

    entropies_deg = [estimate_entropy_deg(samples[:n]) for n in sample_sizes]

    plt.figure()
    plt.plot(sample_sizes, entropies_deg, marker="o", color="orange")
    plt.xscale("log")
    plt.xlabel("Number of samples (log scale)")
    plt.ylabel("Estimated entropy H(p)")
    plt.title("Entropy of GMM with Nearly-Degenerate Weights")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{outdir}/GMM_entropy_degenerate.png", bbox_inches="tight", dpi=150)
    plt.close()

    summary = {
        "weights": weights.tolist(),
        "degenerate_weights": deg_weights.tolist(),
        "means": means.tolist(),
        "covs": covs.tolist(),
        "sample_sizes": sample_sizes,
        "entropy": entropies,
        "entropy_degenerate": entropies_deg,
        "seed": seed
    }
    with open(f"{outdir}/GMM_summary.json","w") as f:
        json.dump(summary, f, indent=2)
