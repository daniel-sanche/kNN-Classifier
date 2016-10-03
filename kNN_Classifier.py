import scipy
from matplotlib import pyplot as plt
import pandas as pd

def plotData(pandasData):
    colors = ["r", "g", "b", "m", "y"]
    for i in range(1,6):
        filtered = pandasData[pandasData["TL"]==i]
        xVals = filtered["x"]
        yVals = filtered["y"]
        plt.scatter(xVals, yVals, c=colors[i-1])
    plt.show()

data = pd.read_csv("knnDataSet.csv")
plotData(data)