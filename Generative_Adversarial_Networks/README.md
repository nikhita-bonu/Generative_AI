Markdown# 2D Generative Adversarial Network (GAN) from Scratch

A PyTorch implementation of a Generative Adversarial Network (GAN) trained on 2D synthetic data distributions. This project focuses on understanding GAN dynamics, adversarial feedback, gradient flow, and mode collapse through controlled experiments rather than standard image datasets.

---

## 📌 Project Overview

This project implements a foundational GAN architecture from scratch to demonstrate core adversarial training mechanics. 

### Key Features
* **Custom Models**: Lightweight PyTorch Generator and Discriminator networks built for 2D coordinate inputs.
* **Synthetic Data Generators**: Programmatically generated single-mode and multi-mode 2D Gaussian distributions.
* **Controlled Gradient Flow**: Precise control over computation graphs using `detach()` during alternating optimization steps.
* **Dynamic Visualizations**: Frame-by-frame sample generation and loss curve plots to monitor distribution alignment over time.

---

## 🎯 Objectives

* Implement Generator and Discriminator architectures using PyTorch.
* Master gradient flow management between networks using `.detach()`.
* Evaluate model behavior on single-modal vs. multi-modal distributions.
* Analyze mode collapse and recovery dynamics.
* Investigate the impact of Discriminator update frequencies on convergence.
* Visualize latent space transformations and training trajectory.

---

## 🛠️ Architecture & Tech Stack

**Tech Stack:** Python, PyTorch, NumPy, Matplotlib, ImageIO

### Model Architectures

#### Generator
Transforms a 2D random noise vector into a synthetic 2D sample point.
Random Noise Vector (Dim: 2)↓Linear (2 → 64)↓LeakyReLU↓Linear (64 → 2)↓Generated Sample (Dim: 2)
#### Discriminator
Evaluates a 2D sample and outputs the probability that it belongs to the real distribution.
Input Sample (Dim: 2)↓Linear (2 → 64)↓LeakyReLU↓Linear (64 → 1)↓Sigmoid↓Real / Fake Probability
---

## 📊 Synthetic Datasets

Rather than relying on image datasets, data points are generated programmatically in `sample.py` to test how effectively the Generator learns explicit modes:

1. **Gaussian (`gaussian`)**: Single 2D Gaussian distribution centered at `[2, 2]` with a variance scale of `1.2`.
2. **Mixture of Two Gaussians (`mixture`)**: Two distinct modes centered at `[2, 2]` and `[-2, -2]`.
3. **Mixture of Four Gaussians (`mixture4`)**: Complex four-corner distribution centered at `[2, 2]`, `[2, -2]`, `[-2, 2]`, and `[-2, -2]` with a variance scale of `1.5`.

---

## ⚙️ Training Details & Gradient Flow

The architecture uses separate Adam optimizers for both networks ($LR = 0.001$).

### Gradient Detachment Strategy
* **Discriminator Step**: Generator samples are detached (`fake_data = G(noise).detach()`) to prevent gradients from updating Generator weights while optimizing the Discriminator.
* **Generator Step**: Generator samples retain gradient tracking (`fake_data = G(noise)`) so backpropagation can route through the Discriminator to update the Generator.

### Default Configuration (`main.py`)
| Parameter | Value |
| :--- | :--- |
| **Training Steps** | 2,500 |
| **Batch Size** | 256 |
| **Data Distribution** | `mixture4` |
| **Generator Learning Rate** | 0.001 |
| **Discriminator Learning Rate** | 0.001 |
| **Latent Vector Dimension** | 2 |
| **Update Ratio (D:G)** | 1 : 1 |
| **Hardware** | CUDA (if available) / CPU |

---

## 🧪 Experiments & Key Insights

1. **Single Gaussian Baseline**: The network easily converged to a single target mode.
2. **Multi-Modal Complexity**: Introducing multi-modal clusters (`mixture` and `mixture4`) produced dynamic adversarial behavior where the Generator experienced temporary mode collapse.
3. **Mode Collapse & Recovery**: Continued training demonstrated that the Generator could dynamically escape partial mode collapse and cover additional modes.
4. **Discriminator Update Frequency**: Adjusting update ratios impacted training stability, highlighting the importance of balanced feedback signals between networks.
5. **Loss Metrics vs. Visuals**: Generator and Discriminator loss curves fluctuated significantly; spatial coordinate visualizations provided a much clearer picture of convergence.

---

## 📁 Directory Structure

```text
.
├── GAN.py          # Generator and Discriminator class definitions
├── train.py        # Discriminator, Generator, and main GAN training functions
├── sample.py       # Synthetic data distribution generators
├── plotting.py     # Loss tracking, frame generation, and visualization utilities
├── main.py         # Entry point for configuration and training execution
├── test/           # Unit and pipeline tests
└── README.md       # Project documentation

Getting Started1. InstallationClone the repository and install dependencies:Bashgit clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
pip install torch numpy matplotlib imageio

2. Running the Training LoopStart training with the default settings (mixture4 distribution):Bashpython main.py
To switch datasets, update the mode parameter inside main.py:Pythonmode = "gaussian"   # Single mode
mode = "mixture"    # Two modes
mode = "mixture4"   # Four modes

 Output ArtifactsTraining visual artifacts are automatically exported:frames/: Contains PNG snapshots of real vs. generated distribution alignment saved every $N$ steps (e.g., frame_00005.png).losses.png: Plot tracking Generator and Discriminator loss trends.2d_gan_frame_by_frame.png: Grid compilation of distribution progress over time.🔮 Future EnhancementsImplement Wasserstein GAN (WGAN) with Gradient Penalty (WGAN-GP).Add Spectral Normalization to linear layers.Explore alternate loss formulations (e.g., Least Squares GAN).Perform systematic hyperparameter grid searches on D:G update ratios.Add quantitative metric evaluations (e.g., Earth Mover's Distance).👤 AuthorNikhita Bonu
