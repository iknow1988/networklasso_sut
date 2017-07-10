import multiprocessing
import numpy as np
import time
from utility import *
from ADMM import *
from loadData import *

def main():
    maxProcesses = multiprocessing.cpu_count()
    lamb = 0
    lambdaMaxValue = 1
    lambdaUpdateStepSize = 0.05
    rho = .0001
    dataSize = 10
    type = 1
    (trainingSet, x_train, y_train, x_test, y_test, trainSize, testSize, sizeOptimizationVariable) = loadData(dataSize, type)
    G1 = generateGraph(60)
    nodes = G1.number_of_nodes()
    edges = G1.number_of_edges()
    print "Number of Nodes = ", nodes, " , Number of Edges = ", edges
    print "Diameter is ", nx.diameter(G1);
    # Initialize ADMM variables
    (A, sqn, sqp, x, u, z) = initializeADMM(G1, sizeOptimizationVariable)
    plot1 = list()
    plot2 = list()
    plot3 = list()
    while(lamb <= lambdaMaxValue or lamb == 0):
        print "For lambda = ", lamb
        start_time = time.time()
        (x, u, z, localVariables) = runADMM(G1, lamb, rho + math.sqrt(lamb), x, u , z, trainingSet, A, sqn, sqp, maxProcesses, sizeOptimizationVariable, trainSize, testSize, x_train, y_train, x_test, y_test)
        print("ADMM finished in %s seconds" % (time.time() - start_time))
        (parameters, trainRMSE) = RRLayer(x, x_train, y_train[:, 0].transpose(), sizeOptimizationVariable - 1)
        testRMSE = getAccuracy(x, parameters, x_test, y_test[:, 0].transpose(), sizeOptimizationVariable - 1)
        plot1.append(trainRMSE)
        plot2.append(testRMSE)
        plot3.append(lamb)
        print "trainRMSE =", trainRMSE, "testRMSE =", testRMSE
        showImageMatrix(x, False, False, '', 'x matrix')
        showImageMatrix(u, False, False, '', 'u matrix')
        showImageMatrix(z, False, False, '', 'z matrix')
        showclusterPerformance(x, False, True, 'train:' + str(trainRMSE) + ' test: ' + str(testRMSE) + ' lamda: ' + str(lamb), "EXP/norm/" + str(lamb) + ".png")
        if(lamb == 0):
            lamb = 0.01
        else:
            lamb = lamb + lambdaUpdateStepSize
    showOverallAccuracy(plot1, plot2, plot3, False, True)

if __name__ == '__main__':
    main()
