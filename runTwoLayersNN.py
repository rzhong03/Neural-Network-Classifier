import os
import time
import numpy as np

# Library for plot the output and save to file
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# Load the CIFAR10 dataset
from keras.datasets import cifar10
baseDir = os.path.dirname(os.path.abspath(__file__)) + '/'
classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xVal = xTrain[49000:, :].astype(np.float)
yVal = np.squeeze(yTrain[49000:, :])
xTrain = xTrain[:49000, :].astype(np.float)
yTrain = np.squeeze(yTrain[:49000, :])
yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)


# Show dimension for each variable
print ('Train image shape:    {0}'.format(xTrain.shape))
print ('Train label shape:    {0}'.format(yTrain.shape))
print ('Validate image shape: {0}'.format(xVal.shape))
print ('Validate label shape: {0}'.format(yVal.shape))
print ('Test image shape:     {0}'.format(xTest.shape))
print ('Test label shape:     {0}'.format(yTest.shape))


# Show some CIFAR10 images
plt.subplot(221)
plt.imshow(xTrain[0])
plt.axis('off')
plt.title(classesName[yTrain[0]])
plt.subplot(222)
plt.imshow(xTrain[1])
plt.axis('off')
plt.title(classesName[yTrain[1]])
plt.subplot(223)
plt.imshow(xVal[0])
plt.axis('off')
plt.title(classesName[yVal[1]])
plt.subplot(224)
plt.imshow(xTest[0])
plt.axis('off')
plt.title(classesName[yTest[0]])
plt.savefig(baseDir+'nn0.png')
plt.clf()


# Normalize the data by subtract the mean image
meanImage = np.mean(xTrain, axis=0)
xTrain -= meanImage
xVal -= meanImage
xTest -= meanImage


# Reshape data from channel to rows
xTrain = np.reshape(xTrain, (xTrain.shape[0], -1))
xVal = np.reshape(xVal, (xVal.shape[0], -1))
xTest = np.reshape(xTest, (xTest.shape[0], -1))
print ('Train image shape after reshape:   {0}'.format(xTrain.shape))
print ('Val image shape after reshape:     {0}'.format(xVal.shape))
print ('Test image shape after reshape:    {0}'.format(xTest.shape))
print ('\n##############################################################################################')


######################################################################################################
#                                       TWO LAYER Neural Network                                     #
######################################################################################################
from twoLayersNN import TwoLayersNN
numClasses = np.max(yTrain) + 1
hiddenNeurons = 100
classifier = TwoLayersNN(xTrain.shape[1], hiddenNeurons, numClasses)


# Show weight of network before training
if classifier.params['w1'] is not None:
    tmpW = classifier.params['w1']
    tmpW = tmpW.reshape(32, 32, 3, hiddenNeurons)
    tmpWMin, tmpWMax = np.min(tmpW), np.max(tmpW)
    for i in range(hiddenNeurons):
        plt.subplot(10, 10, i+1)
        plt.gca().axis('off')
        wPlot = 255.0 * (tmpW[:, :, :, i].squeeze() - tmpWMin) / (tmpWMax - tmpWMin)
        plt.imshow(wPlot.astype('uint8'))
    plt.savefig(baseDir+'nn1.png', bbox_inches='tight')
    plt.clf()


# Training classifier
startTime = time.time()
classifier.train(xTrain, yTrain, lr=5e-3, reg=5e-3, iterations=1500 ,verbose=True)
print ('Training time: {0}'.format(time.time() - startTime))


# Calculate accuracy (Should get around this)
# Training acc:   30.01%
# Validating acc: 29.29%
# Testing acc:    30.18%
print ('Training acc:   {0}%'.format(classifier.calAccuracy(xTrain, yTrain)))
print ('Validating acc: {0}%'.format(classifier.calAccuracy(xVal, yVal)))
print ('Testing acc:    {0}%'.format(classifier.calAccuracy(xTest, yTest)))


# Show weight of network after training
if classifier.params['w1'] is not None:
    tmpW = classifier.params['w1']
    tmpW = tmpW.reshape(32, 32, 3, hiddenNeurons)
    tmpWMin, tmpWMax = np.min(tmpW), np.max(tmpW)
    for i in range(hiddenNeurons):
        plt.subplot(10, 10, i+1)
        plt.gca().axis('off')
        wPlot = 255.0 * (tmpW[:, :, :, i].squeeze() - tmpWMin) / (tmpWMax - tmpWMin)
        plt.imshow(wPlot.astype('uint8'))
    plt.savefig(baseDir+'nn2.png')
    plt.clf()


bestParameters = [0, 0]
bestAcc = -1
bestModel = None
################################################################################
# TODO: 10 points                                                              #
# Tuneup hyper parameters (regularization strength, learning rate)             #
# by using validation set.                                                     #
# - Store the best variables (lr, reg) in bestParameters                       #
# - Store the best model in bestModel                                          #
# - Store the best accuracy in bestAcc                                         #
# - Best Model should get validation accuracy above 35%                        #
################################################################################
print ('\n##############################################################################################')
print('Finding best model......')
learning_rates = [7*1e-3, 9*1e-3, 11*1e-3]
regularization_strengths = [7*3e-3,9*3e-3,11*3e-3]

for lr in learning_rates:
    for reg in regularization_strengths:
        classifier = TwoLayersNN(xTrain.shape[1], hiddenNeurons, numClasses)
        classifier.train(xTrain, yTrain, lr, reg, iterations=1500, verbose=False)
        val_accuracy = classifier.calAccuracy(xVal, yVal)
        if val_accuracy > bestAcc:
            bestAcc = val_accuracy
            bestModel = classifier
        bestParameters = [lr, reg]

################################################################################
#                              END OF YOUR CODE                                #
################################################################################
print ('Best validation accuracy: {0}'.format(bestAcc))


# Predict with best model
if bestModel is not None:
    print ('Best Model parameter, lr = {0}, reg = {1}'.format(bestParameters[0], bestParameters[1]))
    print ('Training acc:   {0}%'.format(bestModel.calAccuracy(xTrain, yTrain)))
    print ('Validating acc: {0}%'.format(bestModel.calAccuracy(xVal, yVal)))
    print ('Testing acc:    {0}%'.format(bestModel.calAccuracy(xTest, yTest)))
######################################################################################################
#                                END OF TWO LAYER Neural Network                                     #
######################################################################################################