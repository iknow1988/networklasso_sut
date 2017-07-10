import math
import numpy as np
from cvxpy import *
from numpy import linalg as LA
import multiprocessing
from multiprocessing import Pool
from utility import *
import time

def solveX(data):
    optimizationVariableSize = int(data[data.size - 1])
    lamb = data[data.size - 2]
    rho = data[data.size - 3]
    sizeData = int(data[data.size - 4])
    trainingSetSize = int(data[data.size - 5])
    x = data[0:optimizationVariableSize]
    trainingData = data[optimizationVariableSize:(optimizationVariableSize + sizeData)]
    neighbors = data[(optimizationVariableSize + sizeData):data.size - 6]
    x_train = trainingData[0:trainingSetSize * optimizationVariableSize]
    y_train = trainingData[trainingSetSize * optimizationVariableSize: trainingSetSize * (optimizationVariableSize + 1)]
    y_train = y_train.reshape(trainingSetSize, 1)
    
    temp = np.zeros((trainingSetSize, optimizationVariableSize))
    for i in range(trainingSetSize):
        temp[i] = x_train[i * optimizationVariableSize:(i + 1) * optimizationVariableSize]
    
    a = Variable(optimizationVariableSize, 1)
    g = 0
    for i in range(trainingSetSize):
        g = g + (temp[i, 0:optimizationVariableSize - 1] * a[0:optimizationVariableSize - 1] + a[optimizationVariableSize - 1] - y_train[i]) ** 2
    
    for i in range(optimizationVariableSize - 1):
        g = g + 0.5 * (square(a[i]))
    
    f = 0
    for i in range(neighbors.size / (2 * optimizationVariableSize + 1)):
        weight = neighbors[i * (2 * optimizationVariableSize + 1)]
        if(weight != 0):
            u = neighbors[i * (2 * optimizationVariableSize + 1) + 1:i * (2 * optimizationVariableSize + 1) + (optimizationVariableSize + 1)]
            z = neighbors[i * (2 * optimizationVariableSize + 1) + (optimizationVariableSize + 1):(i + 1) * (2 * optimizationVariableSize + 1)]
            f = f + rho / 2 * square(norm(a - z + u))
    
    objective = Minimize(50 * g + 50 * f)
    constraints = []
    p = Problem(objective, constraints)
    result = p.solve()
#     if (p._status != "optimal"):
#         print p._status
    return a.value, g.value

def solveZ(data):
    optimizationVariableSize = int(data[data.size - 1])
    lamb = data[data.size - 2]
    rho = data[data.size - 3]
    weight = data[data.size - 4]
    x1 = data[0:optimizationVariableSize]
    x2 = data[optimizationVariableSize:2 * optimizationVariableSize]
    u1 = data[2 * optimizationVariableSize:3 * optimizationVariableSize]
    u2 = data[3 * optimizationVariableSize:4 * optimizationVariableSize]
    a = x1 + u1
    b = x2 + u2
    
    (z1, z2) = (0, 0)
    theta = max(1 - lamb * weight / (rho * LA.norm(a - b) + 0.000001), 0.5)  # So no divide by zero error
    z1 = theta * a + (1 - theta) * b
    z2 = theta * b + (1 - theta) * a
          
    znew = np.matrix(np.concatenate([z1, z2]))
    znew = znew.reshape(2 * optimizationVariableSize, 1)
    return znew

def solveU(data):
    length = data.size
    u = data[0:length / 3]
    x = data[length / 3:2 * length / 3]
    z = data[(2 * length / 3):length]
    
    return u + (x - z)

def getNeighborsModelParameters(G1, maxdeg, u , z, sizeOptimizationVariable):
    nodes = G1.number_of_nodes()
    edges = G1.number_of_edges()

    neighbors = np.zeros(((2 * sizeOptimizationVariable + 1) * maxdeg, nodes))
    edgenum = 0
    numSoFar = {}
    for EI in G1.edges(data=True):
        if not EI[0] in numSoFar:
            numSoFar[EI[0]] = 0
        sourceNode = EI[0]
        neighborIndex = numSoFar[EI[0]]
        neighbors[neighborIndex * (2 * sizeOptimizationVariable + 1), sourceNode] = EI[2]['weight']
        neighbors[neighborIndex * (2 * sizeOptimizationVariable + 1) + 1:neighborIndex * (2 * sizeOptimizationVariable + 1) + (sizeOptimizationVariable + 1), sourceNode] = u[:, 2 * edgenum] 
        neighbors[neighborIndex * (2 * sizeOptimizationVariable + 1) + (sizeOptimizationVariable + 1):(neighborIndex + 1) * (2 * sizeOptimizationVariable + 1), sourceNode] = z[:, 2 * edgenum]
        numSoFar[EI[0]] = numSoFar[EI[0]] + 1

        if not EI[1] in numSoFar:
            numSoFar[EI[1]] = 0
        sourceNode = EI[1]
        neighborIndex = numSoFar[EI[1]]
        neighbors[neighborIndex * (2 * sizeOptimizationVariable + 1), sourceNode] = EI[2]['weight']
        neighbors[neighborIndex * (2 * sizeOptimizationVariable + 1) + 1:neighborIndex * (2 * sizeOptimizationVariable + 1) + (sizeOptimizationVariable + 1), sourceNode] = u[:, 2 * edgenum + 1] 
        neighbors[neighborIndex * (2 * sizeOptimizationVariable + 1) + (sizeOptimizationVariable + 1):(neighborIndex + 1) * (2 * sizeOptimizationVariable + 1), sourceNode] = z[:, 2 * edgenum + 1]
        numSoFar[EI[1]] = numSoFar[EI[1]] + 1
        
        edgenum = edgenum + 1
    
    return neighbors
        
def initializeADMM(G1, sizeOptimizationVariable):
    nodes = G1.number_of_nodes()
    edges = G1.number_of_edges()
    counter = 0
    A = np.zeros((2 * edges, nodes))
    for EI in G1.edges():
        A[2 * counter, EI[0]] = 1
        A[2 * counter + 1, EI[1]] = 1
        counter = counter + 1
    (sqn, sqp) = (math.sqrt(nodes * sizeOptimizationVariable), math.sqrt(2 * sizeOptimizationVariable * edges))
    x = np.zeros((sizeOptimizationVariable, nodes))
    u = np.zeros((sizeOptimizationVariable, 2 * edges))
    z = np.zeros((sizeOptimizationVariable, 2 * edges))
    
    return A, sqn, sqp, x, u, z

def runADMM(G1, lamb, rho, x, u, z, a, A, sqn, sqp, maxProcesses, sizeOptimizationVariable, trainSize, testSize, x_train, y_train, x_test, y_test):
    nodes = G1.number_of_nodes()
    edges = G1.number_of_edges()
    maxdeg = max(G1.degree().values());
    sizeData = a.shape[0]
    # initialize stopping Criterions
    (r, s, epri, edual, counter) = (1, 1, 0, 0, 0)
    iters = 0
    eabs = math.pow(10, -3)
    erel = math.pow(10, -3)
    admmMaxIteration = 50
    pool = Pool(maxProcesses)
    
    # variables
    plots = variables()
    while(iters < admmMaxIteration and (r > epri or s > edual or iters < 1)):
        print "\t At Iteration = ", iters
        start_time = time.time()
        neighbors = getNeighborsModelParameters(G1, maxdeg, u , z, sizeOptimizationVariable)
        params = np.tile([trainSize, sizeData, rho, lamb, sizeOptimizationVariable], (nodes, 1)).transpose()
        temp = np.concatenate((x, a, neighbors, params), axis=0)
        values = pool.map(solveX, temp.transpose())
        newx = np.array(values)[:, 0].tolist()
        x = np.array(newx).transpose()[0]
        
        ztemp = z.reshape(2 * sizeOptimizationVariable, edges, order='F')
        utemp = u.reshape(2 * sizeOptimizationVariable, edges, order='F')
        xtemp = np.zeros((sizeOptimizationVariable, 2 * edges))
        counter = 0
        weightsList = np.zeros((1, edges))
        for EI in G1.edges(data=True):
            xtemp[:, 2 * counter] = np.array(x[:, EI[0]])
            xtemp[:, 2 * counter + 1] = x[:, EI[1]]
            weightsList[0, counter] = EI[2]['weight']
            counter = counter + 1
        xtemp = xtemp.reshape(2 * sizeOptimizationVariable, edges, order='F')
        temp = np.concatenate((xtemp, utemp, ztemp, np.reshape(weightsList, (-1, edges)), np.tile([rho, lamb, sizeOptimizationVariable], (edges, 1)).transpose()), axis=0)
        newz = pool.map(solveZ, temp.transpose())
        ztemp = np.array(newz).transpose()[0]
        ztemp = ztemp.reshape(sizeOptimizationVariable, 2 * edges, order='F')
        s = LA.norm(rho * np.dot(A.transpose(), (ztemp - z).transpose()))  # For dual residual
        z = ztemp
        
        (xtemp, counter) = (np.zeros((sizeOptimizationVariable, 2 * edges)), 0)
        for EI in G1.edges(data=True):
            xtemp[:, 2 * counter] = np.array(x[:, EI[0]])
            xtemp[:, 2 * counter + 1] = x[:, EI[1]]
            counter = counter + 1
        temp = np.concatenate((u, xtemp, z), axis=0)
        newu = pool.map(solveU, temp.transpose())
        u = np.array(newu).transpose()
        
        epri = sqp * eabs + erel * max(LA.norm(np.dot(A, x.transpose()), 'fro'), LA.norm(z, 'fro'))
        edual = sqn * eabs + erel * LA.norm(np.dot(A.transpose(), u.transpose()), 'fro')
        r = LA.norm(np.dot(A, x.transpose()) - z.transpose(), 'fro')
        
        (parameters, trainRMSE) = RRLayer(x, x_train, y_train[:, 0].transpose(), sizeOptimizationVariable - 1)
        testRMSE = getAccuracy(x, parameters, x_test, y_test[:, 0].transpose(), sizeOptimizationVariable - 1)
        normOfOptimizationParameters = LA.norm(x.transpose(), axis=1)
        showclusterPerformance(x, False, False, 'train:' + str(trainRMSE) + ' test: ' + str(testRMSE) + ' iter: ' + str(lamb), "EXP/norm/" + str(iters) + ".png")
        plots['trainRMSE'].append(trainRMSE)
        plots['testRMSE'].append(testRMSE)
        plots['iters'].append(iters)
        plots['normOfOptimizationParameters'].append(normOfOptimizationParameters)
        plots['epri'].append(epri)
        plots['edual'].append(edual)
        plots['r'].append(r)
        plots['s'].append(s)
        plots['x'].append(x)   
        plots['u'].append(u) 
        plots['z'].append(z)
        
        print "\t epri =", epri, "primal resi =", r, "edual =", edual, "dual resi =", s
        print "\t took time ", (time.time() - start_time)
        print "\t train =", trainRMSE, "test =", testRMSE
        iters = iters + 1
    
    pool.close()
    pool.join()
    plotLocalVariables(plots, False, False, False, True)
    
    return (x, u, z, plots)

