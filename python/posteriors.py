import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Tuple, List
import queue

PROB_LABEL = 'p'

# MapData class is defined to store information for heatmaps
class MapData:
    def __init__(self, par1_label: str, par2_label: str, par1_sums: defaultdict, par2_sums: defaultdict):
        self.par1_label = par1_label 
        self.par2_label = par2_label
        self.par1_sums = par1_sums
        self.par2_sums = par2_sums


df = pd.read_csv('results/addm_posteriors.csv')

# num columns - 1 (exclude probability)
N = df.shape[1] - 1

# Calculate the sums of probabilities for each parameter
param_sums: List[Tuple[str, defaultdict]] = [] 
# Example entry: ('d', {0.005: 0.25, 0.006: 0.5, 0.005: 0.025})
for param in df:
    if param != PROB_LABEL:
        param_dict = defaultdict(int)
        for i, row in df.iterrows(): 
            param_row_val = row[param] # Current value of the parameter
            param_dict[param_row_val] += row[PROB_LABEL] # Probability determined given 
                # that parameter and some combination of the other parameters
        param_sums.append((param, param_dict))

# Create pairs of parameters for heatmaps
# Iterate down each column starting with the first parameter
heatmaps: List[MapData] = list()
for i in range(N):
    for j in range(i + 1, N):
        heatmaps.append(MapData(
            param_sums[i][0], param_sums[j][0], 
                param_sums[i][1], param_sums[j][1]))

# Queue to store data for heatmaps
heatmaps_queue: queue.Queue[Tuple[MapData, np.ndarray]] = queue.Queue()

# Iterate over each MapData object and calculate the sums for the 
# corresponding heatmap
for map_data in heatmaps: 
    print(map_data.par1_label, map_data.par2_label, "p_sum")
    # Create empty heatmap data based on number of possible values for 
    # each parameter
    arr_data = np.zeros((len(map_data.par1_sums), len(map_data.par2_sums))) 
    for i in range(len(map_data.par1_sums)):
        for j in range(len(map_data.par2_sums)):
            curr_par1 = list(map_data.par1_sums.keys())[i] # first parameter
            curr_par2 = list(map_data.par2_sums.keys())[j] # second parameter
            # Get the sum of all probabilities for the current pair of 
            # parameter values
            for k, row in df.iterrows():
                if row[map_data.par1_label] == curr_par1 and row[map_data.par2_label] == curr_par2:
                    arr_data[i][j] += row[PROB_LABEL]
            print(curr_par1, curr_par2, arr_data[i][j])
    heatmaps_queue.put((map_data, arr_data))
    print()

# Create subplots grid for the bar plots and heatmaps
fig, axes = plt.subplots(figsize=(10, 10), ncols=N, nrows=N)

# Iterate over each subplot and plot either a bar plot or a heatmap
diag_idx = 0 
for i in range(N):
    for j in range(N):
        if i < j:
            # Turn off axes for subplots above the diagonal 
            axes[i, j].axis('off')  
        elif i == j:
            curr = param_sums[diag_idx]
            vals = curr[1]
            g = sns.barplot(
                x=list(vals.keys()), 
                y=list(vals.values()), 
                ax=axes[i, j], 
                color="grey"
                )
            g.bar_label(g.containers[0], size=8)  

            keys_labels = list(vals.keys())
            axes[i, j].set_xticks(np.arange(len(keys_labels)))
            axes[i, j].set_xticklabels(keys_labels)

            diag_idx += 1
        else:
            curr_heatmap = heatmaps_queue.get()
            data = curr_heatmap[0]
            arr = curr_heatmap[1]
            g = sns.heatmap(
                arr.T, 
                annot=True, 
                cbar=False, 
                cmap="crest", 
                ax=axes[i, j]
                )
            # Invert y-axis for heatmap to match the order of values
            g.invert_yaxis()  
            

            g.set_xticks(np.arange(len(data.par1_sums)) + 0.5, minor=False)  
            g.set_xticklabels(list(data.par1_sums.keys()), minor=False)  
            g.set_yticks(np.arange(len(data.par2_sums)) + 0.5, minor=False)  
            g.set_yticklabels(list(data.par2_sums.keys()), minor=False) 

# Set labels for x and y axes of plot
for i in range(N):
    axes[i, 0].set_ylabel(df.columns[i], size=16)
    axes[N - 1, i].set_xlabel(df.columns[i], size=16)

plt.tight_layout()
plt.show()
