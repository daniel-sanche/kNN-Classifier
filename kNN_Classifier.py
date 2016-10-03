import scipy
from matplotlib import pyplot as plt
import pandas as pd

def plotData(pandasData):
    colors = ["r", "g", "b", "m", "y"]
    for i in range(1,6):
        filtered = pandasData[pandasData["TL"]==i]
        withL = filtered[filtered["L"].notnull()]
        withoutL = filtered[filtered["L"].isnull()]
        plt.scatter(withL["x"], withL["y"], c=colors[i-1], marker="o")
        plt.scatter(withoutL["x"], withoutL["y"], c=colors[i-1], marker="x")

    plt.show()

data = pd.read_csv("knnDataSet.csv")
plotData(data)