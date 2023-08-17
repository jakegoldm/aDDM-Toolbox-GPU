import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from datetime import datetime
import sys

def main():
    data = pd.read_csv("results/ddm_mle.csv")

    # Create a pivot table to reshape the data
    pivot_table = pd.pivot_table(data, values='NLL', index='sigma', columns='d')

    # Convert the pivot table to a numpy array
    heatmap_data = pivot_table.values

    # Create the heatmap
    _, ax = plt.subplots()
    im = ax.imshow(heatmap_data, cmap='cividis', aspect='auto', vmax=6500)

    xlabs = set()
    for row in data['d']:
        xlabs.add(row)
    xlabs = list(xlabs)
    xlabs.reverse()

    ylabs = set()
    for row in data['sigma']:
        ylabs.add(row)
    ylabs = list(ylabs)
    ylabs.sort()

    ax.set_xticks(np.arange(len(xlabs)), labels=xlabs)
    ax.set_yticks(np.arange(len(ylabs)), labels=ylabs)
    _ = ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel('d')
    ax.set_ylabel('sigma')
    ax.set_title('NLL Heatmap')

    if (len(sys.argv) > 1 and "save" in sys.argv):
        currTime = datetime.now().strftime(u"%Y-%m-%d_%H:%M:%S")
        plt.savefig("imgs/ddm_mle_" + currTime + ".png")

    plt.show()


if __name__ == "__main__":
    main()