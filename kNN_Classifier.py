import scipy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import numpy.matlib

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

def kNN(trainData, testData, k=1, feedback_classification=False):
    pointArr = trainData.as_matrix(columns=trainData.columns[0:2])
    pointArr =  np.expand_dims(pointArr, 2)
    labelArr = trainData.as_matrix(columns=trainData.columns[2:])

    testArr = testData.as_matrix()
    testArr = np.expand_dims(testArr, 2)
    testArr = np.transpose(testArr, (2,1,0))

    #find the difference between each point in the test set, and each point in the training set
    diffArr = pointArr - testArr
    #square each cell
    diffArr = np.square(diffArr)
    #add the difference in x and y value together
    #(no need to square this value, because we are looking at relative distances)
    diffArr = np.sum(diffArr, axis=1)
    #combine the diffArr and labelArr, so we can sort and keep the labels
    combined = np.dstack((diffArr, np.matlib.repmat(labelArr, 1, diffArr.shape[1])))
    #sort the array by the 1st dimension to find the smallest distances
    sorted = np.sort(combined, 1)
    #trim the matrix to only keep the top k neighbours
    sorted = sorted[0:k,:,1]
    #find the median of each row to determine the top class for each point
    result = np.median(sorted, axis=0)
    return result

data = pd.read_csv("knnDataSet.csv")
trainData = data[data["L"].notnull()]
testData = data[data["L"].isnull()]

kNN(trainData[['x', 'y', 'L']], testData[['x', 'y']], k=5)