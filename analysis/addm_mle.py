import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from datetime import datetime

def main(): 
    df = pd.read_csv("results/addm_mle.csv")
    thetas = df['theta'].unique()

    num_thetas = len(thetas)
    num_cols = 2  # Number of columns in the subplot grid
    num_rows = (num_thetas - 1) // num_cols + 1  # Number of rows in the subplot grid

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 9))

    vmin = df['NLL'].min()

    for i, theta in enumerate(thetas):
        data = df[df['theta'] == theta]

        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]

        heatmap = ax.imshow(
            np.reshape(data['NLL'], (-1, len(data['d'].unique()))),
            cmap='gist_heat_r',
            extent=[0, len(data['d'].unique()) - 1, 0, len(data['sigma'].unique()) - 1],
            origin='lower',
            aspect='auto',
            vmin=vmin,
            vmax=6200
        )
        ax.set_title(f'theta = {theta}')
        ax.set_xlabel('d')
        ax.set_ylabel('sigma')

        ax.set_xticks(np.arange(len(data['d'].unique())))
        ax.set_xticklabels(data['d'].unique())
        ax.set_yticks(np.arange(len(data['sigma'].unique())))
        ax.set_yticklabels(data['sigma'].unique())

        fig.colorbar(heatmap, ax=ax)

    min_nll_idx = df['NLL'].idxmin()
    min_d = df.loc[min_nll_idx, 'd']
    min_sigma = df.loc[min_nll_idx, 'sigma']
    min_theta = df.loc[min_nll_idx, 'theta']

    annotation = f'Min NLL: d={min_d}, sigma={min_sigma}, theta={min_theta}'
    fig.text(0.5, 0.02, annotation, ha='center', va='center', fontsize=12)

    fig.tight_layout()

    if (len(sys.argv) > 1 and "save" in sys.argv):
        currTime = datetime.now().strftime(u"%Y-%m-%d_%H:%M:%S")
        plt.savefig("imgs/addm_mle_" + currTime + ".png")

    plt.show()


if __name__ == "__main__":
    main()

