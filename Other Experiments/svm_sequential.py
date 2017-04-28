import numpy as np
from cvxpy import *
import matplotlib.pyplot as plt
import time
from sklearn import svm
np.random.seed(2)

def generateSyntheticData(example):
    if(example == 1):
        sizeOptVar = 51
        group = 20
        trainDataSize = 25000
        testDataSize = 10000
        a_true_group = np.random.randn(group, sizeOptVar)
        x_train = np.random.randn(trainDataSize, sizeOptVar)
        y_train = np.zeros((trainDataSize, 1))               
        x_test = np.random.randn(testDataSize, sizeOptVar)
        y_test = np.zeros((testDataSize, 1))
        v_noise_train = np.random.randn(trainDataSize, 1)
        v_noise_test = np.random.randn(testDataSize, 1)
        
        for i in range(trainDataSize):
            a_part = a_true_group[ i / (trainDataSize / group), :]
            x_train[i, sizeOptVar - 1] = 1
            y_train[i] = np.sign([np.dot(a_part.transpose(), x_train[i, 0:sizeOptVar]) + v_noise_train[i]])
        
        for i in range(testDataSize):
            a_part = a_true_group[ i / (testDataSize / group), :]
            x_test[i, sizeOptVar - 1] = 1
            y_test[i] = np.sign([np.dot(a_part.transpose(), x_test[i, 0:sizeOptVar]) + v_noise_test[i]])
        
        return x_train, y_train, x_test, y_test
    else:
        sizeOptVar = 20
        trainDataSize = 1000
        testDataSize = trainDataSize
        DENSITY = 0.2
        beta_true = np.random.randn(sizeOptVar, 1)
        idxs = np.random.choice(range(sizeOptVar), int((1 - DENSITY) * sizeOptVar), replace=False)
        for idx in idxs:
            beta_true[idx] = 0
        offset = 0
        sigma = 45
        x_train = np.random.normal(0, 5, size=(trainDataSize, sizeOptVar))
        y_train = np.sign(x_train.dot(beta_true) + offset + np.random.normal(0, sigma, size=(trainDataSize, 1)))
        x_test = np.random.normal(0, 5, size=(testDataSize, sizeOptVar))
        y_test = np.sign(x_test.dot(beta_true) + offset + np.random.normal(0, sigma, size=(testDataSize, 1)))
        
        return x_train, y_train, x_test, y_test

def problemFormulation(x_train, y_train, a_estimated, v, lambd, method):
    
    if(method == 1):
        loss = sum_entries(pos(1 - mul_elemwise(y_train, x_train * a_estimated - v)))
        reg = norm(a_estimated, 1)
        prob = Problem(Minimize(loss / x_train.shape[0] + lambd * reg))    
        return prob
    elif(method == 2):
        epsil = Variable(x_train.shape[0], 1)
        constraints = [epsil >= 0]
        g = 0.75 * norm(epsil, 1)
        for i in range(x_train.shape[1]):
            g = g + 0.5 * square(a_estimated[i])
        for i in range(x_train.shape[0]):
            constraints = constraints + [y_train[i] * (x_train[i] * a_estimated) >= 1 - epsil[i]]
         
        objective = Minimize(g)
        prob = Problem(objective, constraints)
        return prob
    else:
        epsil = Variable(x_train.shape[0], 1)
        constraints = [epsil >= 0]
        for i in range(x_train.shape[0]):
            constraints = constraints + [y_train[i] * (x_train[i] * a_estimated) >= 1 - epsil[i]]
         
        objective = Minimize(sum_entries(square(a_estimated)) + 0.75 * sum_entries(norm(epsil, 1)))
        prob = Problem(objective, constraints)
        return prob
    
def getAccuracy(a_pred, featureData, labelData, v, error):
    
    if(error == 0):
        # Get accuracy
        (right, total) = (0, featureData.shape[0])
        for j in range(total):
            pred = np.sign([np.dot(a_pred.value.transpose(), featureData[j, :])])
            if(pred == labelData[j]):
                right = right + 1
                
        return right / float(total)
    else:
        return (labelData != np.sign(featureData.dot(a_pred.value) - v.value)).sum() / float(featureData.shape[0])

def plotCurve(lambda_vals, training_accuracy, testing_accuracy, a_estimated_values):
    plotRegularizationPath = 0
    plotAccuracyCurve = 1
    # Plot accuracy curve.
    if(plotAccuracyCurve == 1):
        plt.plot(lambda_vals, training_accuracy, label="Training accuracy")
        plt.plot(lambda_vals, testing_accuracy, label="Test accuracy")
        plt.xscale('log')
        plt.legend(loc='upper left')
        plt.xlabel(r"constant", fontsize=16)
        plt.show()
    
    # Plot the regularization path for a_estimated.
    if(plotRegularizationPath == 1):
        for i in range(a_estimated_values[0].shape[0]):
            plt.plot(lambda_vals, [wi[i, 0] for wi in a_estimated_values])
        plt.xlabel(r"$\lambda$", fontsize=16)
        plt.xscale("log")
        plt.show()

def librarySVM(x_train, y_train, x_test, y_test):
    print "fitting model"
    start_time = time.time()
    model = svm.SVC(kernel='linear', C=0.75, gamma='auto', cache_size=1000, verbose=1) 
    model.fit(x_train, y_train.ravel())
    print("--- %s seconds ---" % (time.time() - start_time))
    print "learning model finished"
    print "Training Accuracy = ", model.score(x_train, y_train)
    print "Testing Accuracy = ", model.score(x_test, y_test)
    print "finish"
    
def main():
    (x_train, y_train, x_test, y_test) = generateSyntheticData(1)

    sizeOptVar = x_train.shape[1]
    trainDataSize = x_train.shape[0]
    testDataSize = x_test.shape[0]
    
    getLibraryResult = 0
    if (getLibraryResult == 1):
        librarySVM(x_train, y_train, x_test, y_test)
    
    method = 2
    TRIALS = 10
    lambda_vals = np.linspace(1, 1, TRIALS)
    if(method == 1):
        lambda_vals = np.logspace(-8, 8, TRIALS)
    training_accuracy = np.zeros(TRIALS)
    testing_accuracy = np.zeros(TRIALS)
    a_estimated_values = []
    for i in range(TRIALS):
        print i
        const = Parameter(sign="positive")
        const.value = lambda_vals[i]
        # const.value = 0.75
        a_estimated = Variable(sizeOptVar)
        v = Variable()
        prob = problemFormulation(x_train, y_train, a_estimated, v, const, method)
        start_time = time.time()
        result = prob.solve()
        print ("result is %f" % result)
        print("--- %s seconds ---" % (time.time() - start_time))
        # Get accuracy
        training_accuracy[i] = getAccuracy(a_estimated, x_train, y_train, v, 0)
        testing_accuracy[i] = getAccuracy(a_estimated, x_test, y_test, v, 0)
        print "Training Accuracy = ", training_accuracy[i]
        print "Testing Accuracy = ", testing_accuracy[i]
        
        a_estimated_values.append(a_estimated.value)
    
    plotCurve(lambda_vals, training_accuracy, testing_accuracy, a_estimated_values)
    
if __name__ == '__main__':
    main()
