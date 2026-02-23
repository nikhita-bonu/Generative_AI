import os, torch, re
from matplotlib import pyplot as plt
import imageio.v2 as imageio
import matplotlib.image as mpimg
import numpy as np
from sample import sample_real_data

max_limit = 8

def plot_generated(G, mode, step, save_dir="frames"):
    os.makedirs(save_dir, exist_ok=True)
    
    G.eval()
    with torch.no_grad():
        z = torch.randn(1000, 2)
        gen_samples = G(z).cpu().numpy()
        real_samples = sample_real_data(1000, mode=mode).cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        ax.scatter(*real_samples.T, alpha=0.3, label='Real')
        ax.scatter(*gen_samples.T, alpha=0.3, label='Generated')
        ax.set_title(f'Step {step}', fontsize=30)

        ax.set_xlim(-max_limit, max_limit)
        ax.set_ylim(-max_limit, max_limit)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        plt.savefig(f"{save_dir}/frame_{step:05d}.png")
        plt.close()
        
    G.train()

def make_video(frame_dir="frames", output_file="gan_training.mp4", fps=10):
    files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
    with imageio.get_writer(output_file, fps=fps) as writer:
        for filename in files:
            img = imageio.imread(os.path.join(frame_dir, filename))
            writer.append_data(img)
    print(f"Video saved to {output_file}")

def plot_losses(losses, plot_every=1000, show=False, output_file='losses.png'):
    num_points = len(losses['D'])
    x = np.arange(plot_every, plot_every * num_points + 1, plot_every)

    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, losses['D'], label='D loss')
    plt.plot(x, losses['G'], label='G loss')
    plt.legend()
    plt.title("Losses")
    plt.xlabel("Step")

    plt.subplot(1, 2, 2)
    plt.plot(x, losses['D(x)'], label='D(x)')
    plt.plot(x, losses['D(G(z))'], label='D(G(z))')
    plt.legend()
    plt.title("Discriminator Outputs")
    plt.xlabel("Step")

    # plt.subplot(1, 3, 3)
    # plt.plot(x, np.array(losses['D(x)']) - np.array(losses['D(G(z))']))
    # plt.title("Adversarial Gap: D(x) - D(G(z))")
    # plt.xlabel("Step")

    plt.tight_layout()
    plt.savefig(output_file)
    if show:
        plt.show()
        
def show_gan_training_grid(
    folder='frames',
    grid_rows=3,
    grid_cols=4,
    filename_pattern=r"frame_(\d+)\.png",
    figsize=(12, 8),
    max_step=None,
    output_file='2d_gan_frame_by_frame.png'
    
):
    frame_files = [f for f in os.listdir(folder) if re.match(filename_pattern, f)]
    steps_and_files = [(int(re.match(filename_pattern, f).group(1)), f) for f in frame_files]

    if max_step is not None:
        steps_and_files = [(s, f) for s, f in steps_and_files if s <= max_step]

    if not steps_and_files:
        raise ValueError("No frames found within the specified max_step.")

    steps_and_files.sort()
    all_steps = [s for s, _ in steps_and_files]
    all_files = [f for _, f in steps_and_files]

    total_steps = all_steps[-1]
    num_frames = grid_rows * grid_cols

    if len(all_steps) < num_frames:
        raise ValueError(f"Not enough frames ({len(all_steps)}) to fill a {grid_rows}×{grid_cols} grid.")

    step_interval = total_steps // (num_frames - 1)
    target_steps = [i * step_interval for i in range(num_frames)]

    selected_files = []
    for ts in target_steps:
        closest_idx = np.argmin([abs(s - ts) for s in all_steps])
        selected_files.append(all_files[closest_idx])

    # --- Plot ---
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=figsize)
    axes = axes.flatten()
    for ax, fname in zip(axes, selected_files):
        img = mpimg.imread(os.path.join(folder, fname))
        step = int(re.match(filename_pattern, fname).group(1))
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()