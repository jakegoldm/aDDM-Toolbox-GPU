import matplotlib.pyplot as plt
import numpy as np
import json

def main():
    with open('results/data.json') as file:
        data = json.load(file)

    rdvs = data["RDVs"]
    x = np.arange(len(rdvs)) * data["timeStep"]
    plt.plot(x, rdvs, label="RDVs")   

    barrier = float(data["barrier"])
    bias = float(data["bias"])
    plt.axhline(barrier, color="grey")
    plt.axhline(-barrier, color="grey")
    plt.axhline(bias, color="grey")

    plt.text(0, -barrier * 0.95, f"d: {round(data['d'], 3)}\n"\
             f"RT: {data['RT']}\nsigma: {round(data['sigma'], 3)}")

    plt.xlabel("Time (ms)")
    plt.ylabel("RDV")
    
    plt.show()

if __name__=="__main__":
    main()