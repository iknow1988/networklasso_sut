import networkx as nx
from cvxpy import *
import numpy as np
from numpy import linalg as LA
import math
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt
from sklearn import mixture

# Input variables
np.random.seed(2)
rho = 0.0001 
eabs = math.pow(10, -2)
erel = math.pow(10, -3)
maxIteration = 50
lambdaThreshold = 50
lamb = 0
lambdaUpdateStepSize = 1.5
nodes = 1000
partitions = 20
sizePart = nodes / partitions
samePartitionEdgeProbability = 0.5
diffPartitionEdgeProbability = 0.01
sizeOptimizationVariable = 51  # Includes 1 for constant offset!
c = 0.75
epsilon = 0.01
trainSetSize = 25
testSetSize = 10
maxProcesses = 4
plot1 = np.zeros(((int(lambdaThreshold / lambdaUpdateStepSize)), 1))
plot2 = np.zeros(((int(lambdaThreshold / lambdaUpdateStepSize)), 1))

def generateGraph():
    G1 = nx.Graph()
    for i in range(nodes):
        G1.add_node(i)
    correctedges = 0
    for NI in G1.nodes():
        for NI2 in G1.nodes():
            if(NI < NI2):
                if ((NI / sizePart) == (NI2 / sizePart)):
                    # Same partition, edge w.p 0.5
                    if(np.random.random() >= 1 - samePartitionEdgeProbability):
                        G1.add_edge(NI, NI2, weight=1)
                        correctedges = correctedges + 1
                else:
                    if(np.random.random() >= 1 - diffPartitionEdgeProbability):
                        G1.add_edge(NI, NI2, weight=1)
    
    return G1

def generateSyntheticData(G1):
    nodes = G1.number_of_nodes()
    edges = G1.number_of_edges()
    a_true = np.random.randn(sizeOptimizationVariable, partitions)
    v = np.random.randn(trainSetSize, nodes)
    vtest = np.random.randn(testSetSize, nodes)

    trainingSet = np.random.randn(trainSetSize * (sizeOptimizationVariable + 1), nodes)  # First all the x_train, then all the y_train below it
    for i in range(trainSetSize):
        trainingSet[(i + 1) * sizeOptimizationVariable - 1, :] = 1  # Constant offset
    for i in range(nodes):
        a_part = a_true[:, i / sizePart]
        for j in range(trainSetSize):
            trainingSet[trainSetSize * sizeOptimizationVariable + j, i] = np.sign([np.dot(a_part.transpose(), trainingSet[j * sizeOptimizationVariable:(j + 1) * sizeOptimizationVariable, i]) + v[j, i]])

    (x_test, y_test) = (np.random.randn(testSetSize * sizeOptimizationVariable, nodes), np.zeros((testSetSize, nodes)))
    for i in range(testSetSize):
        x_test[(i + 1) * sizeOptimizationVariable - 1, :] = 1  # Constant offset
    for i in range(nodes):
        a_part = a_true[:, i / sizePart]
        for j in range(testSetSize):
            y_test[j, i] = np.sign([np.dot(a_part.transpose(), x_test[j * sizeOptimizationVariable:(j + 1) * sizeOptimizationVariable, i]) + vtest[j, i]])
    
    return trainingSet, x_test, y_test, a_true

def getNeighbors(G1, maxdeg, u , z):
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

def solveX(data):
    optimizationVariableSize = int(data[data.size - 1])
    lamb = data[data.size - 2]
    rho = data[data.size - 3]
    sizeData = int(data[data.size - 4])
    trainingSetSize = int(data[data.size - 5])
    c = data[data.size - 6]
    x = data[0:optimizationVariableSize]
    trainingData = data[optimizationVariableSize:(optimizationVariableSize + sizeData)]
    neighbors = data[(optimizationVariableSize + sizeData):data.size - 6]
    x_train = trainingData[0:trainingSetSize * optimizationVariableSize]
    y_train = trainingData[trainingSetSize * optimizationVariableSize: trainingSetSize * (optimizationVariableSize + 1)]
    
    a = Variable(optimizationVariableSize, 1)
    epsil = Variable(trainingSetSize, 1)
    constraints = [epsil >= 0]
    
    g = c * norm(epsil, 1)
    for i in range(optimizationVariableSize - 1):
        g = g + 0.5 * square(a[i])
    for i in range(trainingSetSize):
        temp = np.asmatrix(x_train[i * optimizationVariableSize:(i + 1) * optimizationVariableSize])
        constraints = constraints + [y_train[i] * (temp * a) >= 1 - epsil[i]]
    
    f = 0
    for i in range(neighbors.size / (2 * optimizationVariableSize + 1)):
        weight = neighbors[i * (2 * optimizationVariableSize + 1)]
        if(weight != 0):
            u = neighbors[i * (2 * optimizationVariableSize + 1) + 1:i * (2 * optimizationVariableSize + 1) + (optimizationVariableSize + 1)]
            z = neighbors[i * (2 * optimizationVariableSize + 1) + (optimizationVariableSize + 1):(i + 1) * (2 * optimizationVariableSize + 1)]
            f = f + rho / 2 * square(norm(a - z + u))
    
    objective = Minimize(50 * g + 50 * f)
    p = Problem(objective, constraints)
    result = p.solve()
    
    return a.value, g.value

def solveZ(data):
    optimizationVariableSize = int(data[data.size - 1])
    lamb = data[data.size - 2]
    rho = data[data.size - 3]
    epsilon = data[data.size - 4]
    weight = data[data.size - 5]
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

def initializeADMM(G1):
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

def runADMM(G1, lamb, rho, x, u, z, a, A, sqn, sqp):
    nodes = G1.number_of_nodes()
    edges = G1.number_of_edges()
    maxdeg = max(G1.degree().values());
    sizeData = a.shape[0]
    # stopping Criterion
    (r, s, epri, edual, counter) = (1, 1, 0, 0, 0)   
    
    # Run ADMM
    iters = 0
    pool = Pool(maxProcesses)
    while(iters < maxIteration and (r > epri or s > edual or iters < 1)):
        print "\t \t At Iteration = ", iters
        start_time = time.time()
        
        # x-update
        neighbors = getNeighbors(G1, maxdeg, u , z)
        params = np.tile([c, trainSetSize, sizeData, rho, lamb, sizeOptimizationVariable], (nodes, 1)).transpose()
        temp = np.concatenate((x, a, neighbors, params), axis=0)
        values = pool.map(solveX, temp.transpose())
        newx = np.array(values)[:, 0].tolist()
        x = np.array(newx).transpose()[0]

        # z-update
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
        temp = np.concatenate((xtemp, utemp, ztemp, np.reshape(weightsList, (-1, edges)), np.tile([epsilon, rho, lamb, sizeOptimizationVariable], (edges, 1)).transpose()), axis=0)
        newz = pool.map(solveZ, temp.transpose())
        ztemp = np.array(newz).transpose()[0]
        ztemp = ztemp.reshape(sizeOptimizationVariable, 2 * edges, order='F')
        s = LA.norm(rho * np.dot(A.transpose(), (ztemp - z).transpose()))  # For dual residual
        z = ztemp
        
        # u-update
        (xtemp, counter) = (np.zeros((sizeOptimizationVariable, 2 * edges)), 0)
        for EI in G1.edges(data=True):
            xtemp[:, 2 * counter] = np.array(x[:, EI[0]])
            xtemp[:, 2 * counter + 1] = x[:, EI[1]]
            counter = counter + 1
        temp = np.concatenate((u, xtemp, z), axis=0)
        newu = pool.map(solveU, temp.transpose())
        u = np.array(newu).transpose()
        
        # Stopping criterion - p19 of ADMM paper
        epri = sqp * eabs + erel * max(LA.norm(np.dot(A, x.transpose()), 'fro'), LA.norm(z, 'fro'))
        edual = sqn * eabs + erel * LA.norm(np.dot(A.transpose(), u.transpose()), 'fro')
        r = LA.norm(np.dot(A, x.transpose()) - z.transpose(), 'fro')
        print "\t \t \t Iteration ", iters, " took time ", (time.time() - start_time), "seconds", "And Primal residual = ", r, " , Dual Residual = ", s
        iters = iters + 1
        
    pool.close()
    pool.join()
    
    return (x, u, z)

def getAccuracy(modelParameters, dataSetSize, featureData, labelData):
    
    (right, total) = (0, dataSetSize * nodes)
    a_pred = modelParameters
    for i in range(nodes):
        temp = a_pred[:, i]
        for j in range(dataSetSize):
            pred = np.sign([np.dot(temp.transpose(), featureData[j * sizeOptimizationVariable:(j + 1) * sizeOptimizationVariable, i])])
            if(pred == labelData[j, i]):
                right = right + 1
    
    return right / float(total)
    
def main():
    start_time = time.time()
    G1 = generateGraph()
    print("----- Graph creation %s seconds -----" % (time.time() - start_time))
    (trainingSet, x_test, y_test, a_true) = generateSyntheticData(G1)
    
    x_train = trainingSet[0:trainSetSize * sizeOptimizationVariable, :]
#     dpgmm = mixture.BayesianGaussianMixture(n_components=25, weight_concentration_prior_type='dirichlet_process',init_params="random", max_iter=100)
#     dpgmm.fit(x_train)
#     clusters = dpgmm.predict(x_train)
    y_train = trainingSet[trainSetSize * sizeOptimizationVariable: trainSetSize * (sizeOptimizationVariable + 1), :]
    sizeData = trainingSet.shape[0]
    nodes = G1.number_of_nodes()
    edges = G1.number_of_edges()
    print "\t Number of Nodes = ", nodes, " , Number of Edges = ", edges
    print "\t Diameter is ", nx.diameter(G1);
    
    # Initialize ADMM variables
    (A, sqn, sqp, x, u, z) = initializeADMM(G1)
    print "Run ADMM algorithm for", int(lambdaThreshold / lambdaUpdateStepSize), "lambda values and process size is", maxProcesses
    count = 0
    global lamb
    while(lamb <= lambdaThreshold or lamb == 0):
        # run ADMM
        print "\t For lambda = ", lamb
        start_time = time.time()
        (x, u, z) = runADMM(G1, lamb, rho + math.sqrt(lamb), x, u , z, trainingSet, A, sqn, sqp)
        print("\t \t ADMM finished in %s seconds" % (time.time() - start_time))
        
#         print "True Variance"
#         var1 = np.var(a_true)
#         var2 = np.var(x)
#         print var1
#         print var2
#         print var1-var2
                
        # Get accuracy
        print "\t \t Result"
        print "\t \t \t Train Data Accuracy = ", getAccuracy(x, trainSetSize, x_train, y_train)
        testAccuracy = getAccuracy(x, testSetSize, x_test, y_test)
        print "\t \t \t Test Data Accuracy = ", testAccuracy
        plot1[count] = lamb
        plot2[count] = testAccuracy
        if(lamb == 0):
            lamb = 0.5
        else:
            lamb = lamb * lambdaUpdateStepSize
        count = count + 1
        
    pl1 = np.array(plot1)
    pl2 = np.array(plot2)
    plt.plot(pl1, pl2)
    plt.xscale('log')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Prediction Accuracy')
    plt.show()
if __name__ == '__main__':
    main()
