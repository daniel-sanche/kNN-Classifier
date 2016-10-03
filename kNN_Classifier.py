import scipy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

def plotData(pandasData, pdf_name="plot.pdf", display_figure=True):
    colors = ["r", "g", "b", "m", "y"]
    fig = plt.figure()
    for i in range(1,6):
        filtered = pandasData[pandasData["TL"]==i]
        withL = filtered[filtered["L"].notnull()]
        withoutL = filtered[filtered["L"].isnull()]
        plt.scatter(withoutL["x"], withoutL["y"], c=colors[i-1], marker="x")
        plt.scatter(withL["x"], withL["y"], c=colors[i-1], marker="o")
    if display_figure:
        plt.show()
    pp = PdfPages(pdf_name)
    pp.savefig(fig)
    pp.close()

data = pd.read_csv("knnDataSet.csv")
plotData(data)