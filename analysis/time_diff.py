import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 99.56% improvement
# python avg: 1087 seconds
# C++ CPU avg: 4.81 seconds

def main():
    py_time = 1087

    df = pd.read_csv("results/test_addm.csv")
    times = df["time"]
    avg = np.mean(times)

    xaxis = ("C++ CPU", "Python")
    yaxis = (avg, py_time)

    print(avg)

    plt.bar(xaxis, yaxis, align="center")
    plt.xticks(np.arange(len(xaxis)), xaxis)
    plt.ylabel("Time (s)")
    plt.title("aDDM MLE Performance Comparison")

    plt.show()

if __name__ == "__main__":
    main()
