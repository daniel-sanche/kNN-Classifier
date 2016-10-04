from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from scipy import stats

#This function is used to render the data to a pdf file, and optionally display it in a figure window
#trainData = the data that needs to be classified
#testData = the data that we know the ground truth for
#classifiedLabels = the labels assigned to trainData. If None, the ground truth class will be plotted instead
#pdf_name = the name of the pdf file to save
#display_figure = whether the figure window should be displayed along with the rendered pdf
def plotData(trainData, testData, classifiedLabels=None, pdf_name="plot.pdf", display_figure=True):
    #define the colors we will use
    colors = ["r", "y", "g", "b", "m"]
    fig = plt.figure()
    for i in range(1, 6):
        #for each class, find the train and test data belonging to the class
        filteredTrain = trainData[trainData["TL"]==i]
        #if we have our own classified labels to show, load those. Otherwise, load the ground truth
        if classifiedLabels is not None:
            filteredTest = testData[classifiedLabels == i]
        else:
            filteredTest = testData[testData["TL"] == i]
        #plot on the figure, with different parkers for test/train values
        plt.scatter(filteredTest["x"], filteredTest["y"], c=colors[i - 1], marker="x")
        plt.scatter(filteredTrain["x"], filteredTrain["y"], c=colors[i - 1], marker="o")
    #display figure if nexessary, and save to pdf file
    if display_figure:
        plt.show()
    pp = PdfPages(pdf_name)
    pp.savefig(fig)
    pp.close()

#This function is used to display a confusion matrix to the console
#labeledClasses = the labels assigned by our algorithm
#groundTruth = the ground truth labels for each point
#titleStr = the title to be displyed above the confusion matrix
def printConfusionMatrix(labeledClasses, groundTruth, titleStr="Confusion Matrix:"):
    #convert to uint8 for consistency
    labeledClasses = labeledClasses.astype(np.uint8)
    groundTruth = groundTruth.astype(np.uint8)
    #initialize values
    num_classes = np.amax(groundTruth)
    confusionMat = np.zeros([num_classes, num_classes], dtype=int)
    totalNum = 0
    totalCorrect = 0

    for i in range(0, labeledClasses.shape[0]):
        #for each label, update the confusion matrix for how we did
        truth = groundTruth[i,0].astype(np.uint8)
        guess = labeledClasses[i].astype(np.uint8)
        confusionMat[truth-1, guess-1] += 1
        totalNum += 1
        if truth == guess:
            totalCorrect += 1
    #display matrix to screen
    print(titleStr, end="\n\t\t")
    for i in range(0,num_classes):
        print(i, end="\t\t")
    for i in range(0,num_classes):
        print("\n"+str(i)+"\t\t", end="")
        for j in range(0,num_classes):
            print(confusionMat[i,j], end="\t\t")
    #display final classification score
    print ("\n"+str(totalCorrect)+"/"+str(totalNum)+" = "+"%.2f" % round((totalCorrect/totalNum*100),2)+"%\n")

#This function will do the classification using the kNearestNeighbours algorithm
#trainData = the data that needs to be classified
#testData = the data that we know the ground truth for
#feedback_classification = whether or now trainData will be added to testData after a label is assigned
def kNN(trainData, testData, k=1, feedback_classification=False):
    #break out the data we need
    pointArr = trainData.as_matrix(columns=trainData.columns[0:2])
    labelArr = trainData.as_matrix(columns=trainData.columns[2:]).astype(np.uint8)
    testArr = testData.as_matrix()
    results = np.zeros(testArr.shape[0])
    #for each point in the training set, attempt to assign it a class
    for i in range(0, testArr.shape[0]):
        #find the next point in the training set
        thisPt = testArr[i,:]
        # find the difference between each point in the test set and the point we are classifying
        distanceMat = pointArr - thisPt
        #square the x and y differences
        distanceMat = np.square(distanceMat)
        # add the difference in x and y value together
        # (no need to square this value, because we are looking at relative distances)
        distanceMat = np.sum(distanceMat, axis=1)
        #find the indices of the sorted list, so that we know which points are closest
        sort_indices = np.argsort(distanceMat)
        #sort the label array using these indices to find the top labels
        closestLabels = labelArr[sort_indices]
        closestLabels = closestLabels[0:k]
        #take the mode of the top k results to find the class
        topResult = stats.mode(closestLabels)[0]
        #store it in the array of other results
        results[i] = topResult
        if feedback_classification:
            #update the training data if requested
            pointArr = np.append(pointArr, thisPt.reshape(1,2), axis=0)
            labelArr = np.append(labelArr, topResult.reshape(1,1), axis=0)
    return results


#read the dataset
data = pd.read_csv("knnDataSet.csv")
#break it into two groups: those with known labels, and those we must classify
trainData = data[data["L"].notnull()]
testData = data[data["L"].isnull()]

#PART 1: Plot and save the ground truth scatter plot for the input data
plotData(trainData, testData, display_figure=False, pdf_name="GroundTruth.pdf")

#PART 3: Classify, plot, and display the confusion matrix where the training data does not update
testStr = "NoDataUpdate"
for k in [1,5,10,20]:
    thisTestStr = str(k)+"K_"+testStr
    newLabels = kNN(trainData[['x', 'y', 'L']], testData[['x', 'y']], k=k, feedback_classification=False)
    printConfusionMatrix(newLabels, testData[['TL']].as_matrix(), titleStr="Confusion Matrix For " +thisTestStr + ":")
    plotData(trainData, testData, classifiedLabels=newLabels, pdf_name=thisTestStr+".pdf", display_figure=False)

#PART 4: Classify, plot, and display the confusion matrix where the training data is sorted by TL, and training data is updated over time
testStr = "SortedUpdate"
for k in [1,5,10]:
    testData = testData.sort_values(["TL"],ascending=True)
    thisTestStr = str(k)+"K_"+testStr
    newLabels = kNN(trainData[['x', 'y', 'L']], testData[['x', 'y']], k=k, feedback_classification=True)
    printConfusionMatrix(newLabels, testData[['TL']].as_matrix(), titleStr="Confusion Matrix For " +thisTestStr + ":")
    plotData(trainData, testData, classifiedLabels=newLabels, pdf_name=thisTestStr+".pdf", display_figure=False)

#PART 5: Classify, plot, and display the confusion matrix where the training data is randomized, and training data is updated over time
testStr = "RandomUpdate"
for k in [1,5,10]:
    testData = testData.sample(frac=1)
    thisTestStr = str(k)+"K_"+testStr
    newLabels = kNN(trainData[['x', 'y', 'L']], testData[['x', 'y']], k=k, feedback_classification=True)
    printConfusionMatrix(newLabels, testData[['TL']].as_matrix(), titleStr="Confusion Matrix For " +thisTestStr + ":")
    plotData(trainData, testData, classifiedLabels=newLabels, pdf_name=thisTestStr+".pdf", display_figure=False)