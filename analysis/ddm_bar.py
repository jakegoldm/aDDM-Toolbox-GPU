import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import sys

def main():
    data = pd.read_csv("results/ddm_likelihoods.csv")

    # Create a pivot table to reshape the data
    pivot_table = pd.pivot_table(data, values='p', index='sigma', columns='d')

    # Convert the pivot table to a numpy array
    heatmap_data = pivot_table.values

    # Get the min and max values of the heatmap data for normalization
    vmin = heatmap_data.min()
    vmax = heatmap_data.max()

    # Prepare the data for 3D bar plot
    x = np.arange(len(pivot_table.columns))
    y = np.arange(len(pivot_table.index))
    x, y = np.meshgrid(x, y)
    z = heatmap_data

    # Create a new colormap with enough colors to map each value in heatmap_data
    cmap = plt.cm.get_cmap('cividis', len(np.unique(heatmap_data)))

    norm = plt.Normalize(vmin=vmin, vmax=0.019)
    colors = cmap(norm(z.flatten()))
    colors = colors.reshape(z.shape + (4,))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(x)):
        for j in range(len(y)):
            ax.bar3d(x[i, j], y[i, j], 0, 0.8, 0.8, z[i, j], color=colors[i, j])

    xlabs = set(data['d'])
    xlabs = list(xlabs)
    xlabs.reverse()

    ylabs = set(data['sigma'])
    ylabs = list(ylabs)
    ylabs.sort()

    ax.set_xticks(np.arange(len(xlabs)), labels=xlabs)
    ax.set_yticks(np.arange(len(ylabs)), labels=ylabs)
    ax.set_xlabel('d')
    ax.set_ylabel('sigma')
    ax.set_zlabel('p')
    ax.set_title('3D Bar Plot of likelihoods')

    if (len(sys.argv) > 1 and "save" in sys.argv):
        currTime = datetime.now().strftime(u"%Y-%m-%d_%H:%M:%S")
        plt.savefig("imgs/ddm_mle_" + currTime + ".png")

    plt.show()

if __name__ == "__main__":
    main()
