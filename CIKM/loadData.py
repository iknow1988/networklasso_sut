import pickle
import numpy as np
import linecache

def prepareDataset(trainingSet, x_test, y_test, trainSize, testSize, sizeOptimizationVariable):
    x_train = trainingSet[0:trainSize * sizeOptimizationVariable, :]
    y_train = trainingSet[trainSize * sizeOptimizationVariable: trainSize * (sizeOptimizationVariable + 1), :]
    temp = np.zeros((trainSize, 60, (sizeOptimizationVariable - 1)))
    for i in range(trainSize):
        temp[i, :] = x_train[i * (sizeOptimizationVariable):i * (sizeOptimizationVariable) + (sizeOptimizationVariable - 1), :].transpose()
    x_train = temp
    temp = np.zeros((testSize, 60, (sizeOptimizationVariable - 1)))
    for i in range(testSize):
        temp[i, :] = x_test[i * (sizeOptimizationVariable):i * (sizeOptimizationVariable) + (sizeOptimizationVariable - 1), :].transpose()
    x_test = temp
    
    return x_train, y_train, x_test

def loadData(dataSize, type):
    if type == 1:
        (trainingSet, x_train, y_train, x_test, y_test, trainSize, testSize, sizeOptimizationVariable) = loadGroupedFeature(dataSize)
    if type == 2:
        fileName = "train_" + str(dataSize) + ".txt"
        print "Data loading from:", fileName
        (trainingSet, x_train, y_train, x_test, y_test, trainSize, testSize, sizeOptimizationVariable) = loadNearbyRadarFeaturesofSite(dataSize, fileName)
    if type == 3:
        fileName = "train_" + str(dataSize) + ".txt"
        print "Data loading from:", fileName
        (trainingSet, x_train, y_train, x_test, y_test, trainSize, testSize, sizeOptimizationVariable) = loadRawData(dataSize, fileName)
    return trainingSet, x_train, y_train, x_test, y_test, trainSize, testSize, sizeOptimizationVariable   

def loadGroupedFeature(dataSize):
    labels = pickle.load(open("10000_labels.p", "rb")).astype(float)
    sizeOptimizationVariable = 23
    nodes = 60
    time = 15
    height = 4
    trainingPercentage = 80
    data = pickle.load(open("RadarData.p", "rb"))
    rawData = np.zeros((10000, time, height, (sizeOptimizationVariable - 1))).astype(np.int16)
    for key in data:
        showerData = key
        for key, histograms in showerData.items():
            rainAmount = key
            data = list()
            for key, value in histograms.items():
                radar = key
                rawData[radar] = value[:, :, 0:(sizeOptimizationVariable - 1)]

    trainSize = int((trainingPercentage / 100.0) * dataSize)
    testSize = int(((100 - trainingPercentage) / 100.0) * dataSize)
    trainingSet = np.zeros((trainSize * (sizeOptimizationVariable + 1), nodes))
    x_test = np.zeros((testSize * sizeOptimizationVariable, nodes))
    y_test = np.zeros((testSize, nodes))
    print "Number of samples at each node ", dataSize
    
    for i in range(dataSize):
        index = i
        if(i < trainSize):
            trainingSet[(index + 1) * sizeOptimizationVariable - 1, :] = 1  # Constant offset
        else:
            index = i - trainSize
            x_test[(index + 1) * sizeOptimizationVariable - 1, :] = 1  # Constant offset
        count = 0
        for t in range(time):
            for h in range(height):
                data = rawData[i, t, h]
                if(i < trainSize):
                    trainingSet[index * (sizeOptimizationVariable):index * (sizeOptimizationVariable) + (sizeOptimizationVariable - 1), count] = data
                    trainingSet[trainSize * (sizeOptimizationVariable) + index, count] = float(labels[i])
                else:
                    x_test[index * (sizeOptimizationVariable):index * (sizeOptimizationVariable) + (sizeOptimizationVariable - 1), count] = data
                    y_test[index, count] = float(labels[i])
                count = count + 1
    (x_train, y_train, x_test) = prepareDataset(trainingSet, x_test, y_test, trainSize, testSize, sizeOptimizationVariable)
    
    return trainingSet, x_train, y_train, x_test, y_test, trainSize, testSize, sizeOptimizationVariable

def loadNearbyRadarFeaturesofSite(dataSize, fileName):
    sizeOptimizationVariable = (9 + 1)
    trainSize = int((80 / 100.0) * dataSize)
    testSize = int((20 / 100.0) * dataSize)
    trainingSet = np.zeros((trainSize * (sizeOptimizationVariable + 1), 60))
    x_test = np.zeros((testSize * sizeOptimizationVariable, 60))
    y_test = np.zeros((testSize, 60))
    print "\tNumber of data at node ", dataSize
    for i in range(dataSize):
        print "\tReading", i + 1
        line = (linecache.getline(fileName, i + 1)).split(',')
        index = i
        if(i < trainSize):
            trainingSet[(index + 1) * sizeOptimizationVariable - 1, :] = 1  # Constant offset
        else:
            index = i - trainSize
            x_test[(index + 1) * sizeOptimizationVariable - 1, :] = 1  # Constant offset
        nodeCount = 0
        for t in range(15):
            for h in range(4):
                TH_ind = t * 4 + h
                img_line = np.asarray(line[2].strip().split(' ')[TH_ind * 101 ** 2:(TH_ind + 1) * 101 ** 2]).astype(np.int)
                img_line = img_line.reshape([101, 101])
                data = np.zeros((9))
                count = 0
                for i1 in range(49, 52):
                    for j1 in range(49, 52):
                        data[count] = img_line[i1, j1]
                        count = count + 1
                if(i < trainSize):
                    trainingSet[index * (sizeOptimizationVariable):index * (sizeOptimizationVariable) + (sizeOptimizationVariable - 1), nodeCount] = data
                    trainingSet[trainSize * (sizeOptimizationVariable) + index, nodeCount] = float(line[1])
                else:
                    x_test[index * (sizeOptimizationVariable):index * (sizeOptimizationVariable) + (sizeOptimizationVariable - 1), nodeCount] = data
                    y_test[index, nodeCount] = float(line[1])
                nodeCount = nodeCount + 1
    (x_train, y_train, x_test) = prepareDataset(trainingSet, x_test, y_test, trainSize, testSize, sizeOptimizationVariable)
    
    return trainingSet, x_train, y_train, x_test, y_test, trainSize, testSize, sizeOptimizationVariable

def loadRawData(dataSize, fileName):
    sizeOptimizationVariable = (101 * 101 + 1)
    trainSize = int((80 / 100.0) * dataSize)
    testSize = int((20 / 100.0) * dataSize)
    trainingSet = np.zeros((trainSize * (sizeOptimizationVariable + 1), 60))
    x_test = np.zeros((testSize * sizeOptimizationVariable, 60))
    y_test = np.zeros((testSize, 60))
    print "\tNumber of data at node ", dataSize
    for i in range(dataSize):
        print "\tReading", i + 1
        line = (linecache.getline(fileName, i + 1)).split(',')
        index = i
        if(i < trainSize):
            trainingSet[(index + 1) * sizeOptimizationVariable - 1, :] = 1  # Constant offset
        else:
            index = i - trainSize
            x_test[(index + 1) * sizeOptimizationVariable - 1, :] = 1  # Constant offset
        nodeCount = 0
        for t in range(15):
            for h in range(4):
                TH_ind = t * 4 + h
                img_line = np.asarray(line[2].strip().split(' ')[TH_ind * 101 ** 2:(TH_ind + 1) * 101 ** 2]).astype(np.int)
                if(i < trainSize):
                    trainingSet[index * (sizeOptimizationVariable):index * (sizeOptimizationVariable) + (sizeOptimizationVariable - 1), nodeCount] = img_line
                    trainingSet[trainSize * (sizeOptimizationVariable) + index, nodeCount] = float(line[1])
                else:
                    x_test[index * (sizeOptimizationVariable):index * (sizeOptimizationVariable) + (sizeOptimizationVariable - 1), nodeCount] = img_line
                    y_test[index, nodeCount] = float(line[1])
                nodeCount = nodeCount + 1
    
    (x_train, y_train, x_test) = prepareDataset(trainingSet, x_test, y_test, trainSize, testSize, sizeOptimizationVariable)
    
    return trainingSet, x_train, y_train, x_test, y_test, trainSize, testSize, sizeOptimizationVariable

