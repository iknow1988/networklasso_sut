import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import mean_squared_error
from sklearn import  linear_model

def showImageMatrix(x, save, show, dir, title):
    if(save or show):
        plt.imshow(x, aspect='auto', interpolation='none', origin='lower')
        plt.colorbar()
        plt.title(title)
    if save:
        plt.savefig("EXP/" + str(iters) + ".png")
        plt.clf()
    if show :
        plt.show()
        plt.clf()
              
def showclusterPerformance(x, save, show, title, dir):
    if(save or show):
        fig, ax = plt.subplots()
        ax.imshow(x, aspect='auto', interpolation='none', origin='lower')
        for (j, i), label in np.ndenumerate(x):
            if (i % 15 == 0):
                label = 'x'
                ax.text(i, j, label, ha='center', va='center', color='black', fontweight='heavy')
        ax.set_xticks(np.arange(0, 60))
        ax.set_yticks(np.arange(0, 16))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(title)
        ax.grid()
        if save:
            plt.savefig(dir)
        if show:
            plt.show()
        plt.clf()
 
def plotLocalVariables(plots, accuracy, residuals , save, show):
    pl1 = np.array(plots['trainRMSE'])
    pl2 = np.array(plots['testRMSE'])
    pl3 = np.array(plots['iters'])
    pl4 = np.array(plots['normOfOptimizationParameters'])
    pl5 = np.array(plots['epri'])
    pl6 = np.array(plots['edual'])
    pl7 = np.array(plots['r'])
    pl8 = np.array(plots['s'])
    x = np.array(plots['x'])
    u = np.array(plots['u'])
    z = np.array(plots['z'])
    
    if residuals:
        plt.plot(pl3, pl5, label="epri") 
        plt.plot(pl3, pl6, label="edual")
        plt.plot(pl3, pl7, label="primal residual") 
        plt.plot(pl3, pl8, label="dual residual")
        plt.legend(loc=0)
        plt.xlabel('iterations')
        plt.ylabel('Residuals')
        if save :
            plt.savefig("EXP/iterations" + ".png")   
            plt.clf()
        if show:
            plt.show()
            plt.clf()
    
    if accuracy:
        plt.plot(pl3, pl1, label="Train Dataset") 
        plt.plot(pl3, pl2, label="Test Dataset")
        plt.legend(loc=0)
        plt.xlabel('iterations')
        plt.ylabel('RMSE')
        if save :
            plt.savefig("EXP/iterations" + ".png")   
            plt.clf()
        if show:
            plt.show()
            plt.clf()

def variables():
    plots = dict()
    plots['trainRMSE'] = list()
    plots['testRMSE'] = list()
    plots['iters'] = list()
    plots['normOfOptimizationParameters'] = list()
    plots['epri'] = list()
    plots['edual'] = list()
    plots['r'] = list()
    plots['s'] = list()
    plots['x'] = list()
    plots['u'] = list()
    plots['z'] = list()
    
    return plots

def generateGraph(nodes):
    G1 = nx.Graph()
    for i in range(nodes):
        G1.add_node(i)
    
    for NI in range(nodes):
        NI2 = NI + 1
        NI3 = NI + 15
        if(NI % 15 == 14):
            if(NI3 < 60):
                G1.add_edge(NI, NI3, weight=1)
        else:
            G1.add_edge(NI, NI2, weight=1)
            if(NI3 < 60):
                G1.add_edge(NI, NI3, weight=1)
    
    return G1

def getAccuracy(networkLassoModels, ridgeRegModels, featureData, labelData, sizeOptimizationVariable):
    networkLassoModels = networkLassoModels.transpose()
    y_pred_1 = np.zeros((labelData.shape[0], 60))
    y_pred = np.zeros((labelData.shape[0], 1))
    for i in range(labelData.shape[0]):
        for j in range(60):
            t = 0
            for k in range(sizeOptimizationVariable):
                t = t + featureData[i, j, k] * networkLassoModels[j, k]
            y_pred_1[i, j] = t + networkLassoModels[j, (sizeOptimizationVariable - 1)]
    
    for i in range(labelData.shape[0]):
        t = 0
        for j in range(60):
            t = t + y_pred_1[i, j] * ridgeRegModels[j]
        y_pred[i] = t     
    
    RMSE = mean_squared_error(labelData, y_pred) ** 0.5   
        
    return RMSE

def RRLayer(modelParameters, featureData, labelData, sizeOptimizationVariable):
    modelParameters = modelParameters.transpose()
    y_pred = np.zeros((labelData.shape[0], 60))
    for i in range(labelData.shape[0]):
        for j in range(60):
            t = 0
            for k in range(sizeOptimizationVariable):
                t = t + featureData[i, j, k] * modelParameters[j, k]
            y_pred[i, j] = t + modelParameters[j, (sizeOptimizationVariable)]
    
    x_train = y_pred
    y_train = labelData
    regr = linear_model.Ridge(alpha=0.01, normalize=True)
    regr.fit(x_train, y_train)
    train_error = mean_squared_error(labelData, y_pred[:, 0]) ** 0.5
    
    return regr.coef_, train_error

def showOverallAccuracy(plot1, plot2, plot3, save, show):
    pl1 = np.array(plot1)
    pl2 = np.array(plot2)
    pl3 = np.array(plot3)
    if(save or show):
        plt.plot(pl3, pl1, label="Train Dataset") 
        plt.plot(pl3, pl2, label="Test Dataset")
        plt.legend()
        plt.xlabel('$\lambda$')
        plt.ylabel('RMSE')
        if save:
            plt.savefig("RMSE_"+str(dataSize) + ".png")
            plt.clf()
        if show:
            plt.show()   
            plt.clf()        