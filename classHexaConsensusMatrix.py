'''
Created on 2016/11/08

@author: Naoya Fujita
'''
import numpy as np
import pandas as pd
import numpy.matlib

#import fastcluster
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import cophenet


class HexaConsensusMatrix(object):
    '''
    Connectivity matrix (single JNMF run)
    Consensus matrix (multiple JNMF run)
    Reordring consensus matrix
    Cophenetic correlation
    
    '''


    def __init__(self, X1, X2, X3, X4, X5, X6):
        '''
        Constructor
        '''
        self.cmW = pd.DataFrame(np.zeros((X1.shape[0], X1.shape[0])), index = X1.index, columns = X1.index)
        self.cmH1 = pd.DataFrame(np.zeros((X1.shape[1], X1.shape[1])), index = X1.columns, columns = X1.columns)
        self.cmH2 = pd.DataFrame(np.zeros((X2.shape[1], X2.shape[1])), index = X2.columns, columns = X2.columns)
        self.cmH3 = pd.DataFrame(np.zeros((X3.shape[1], X3.shape[1])), index = X3.columns, columns = X3.columns)
        self.cmH4 = pd.DataFrame(np.zeros((X4.shape[1], X4.shape[1])), index = X4.columns, columns = X4.columns)
        self.cmH5 = pd.DataFrame(np.zeros((X5.shape[1], X5.shape[1])), index = X5.columns, columns = X5.columns)
        self.cmH6 = pd.DataFrame(np.zeros((X6.shape[1], X6.shape[1])), index = X6.columns, columns = X6.columns)
        sizePanH = X1.shape[1] + X2.shape[1] + X3.shape[1] + X4.shape[1] + X5.shape[1] + X6.shape[1]
        columnsPanH = ["X1_" + str(x) for x in X1.columns] + ["X2_" + str(x) for x in X2.columns] + ["X3_" + str(x) for x in X3.columns] + ["X4_" + str(x) for x in X4.columns] + ["X5_" + str(x) for x in X5.columns] + ["X6_" + str(x) for x in X6.columns]
        self.cmPanH = pd.DataFrame(np.zeros((sizePanH, sizePanH)), index = columnsPanH, columns = columnsPanH)
        self.cmNum = 0
    #this　function is designed for calculateing the value in consensus matrix W
    def calcConnectivityW(self, W):
        maxW = W.to_numpy().max(axis=1)
        #change to_numpy = to_numpy
        maxW[maxW == 0] = 1
        # argmaxW = W.to_numpy().argmax(axis=1)
        maxMatW = (np.tile(maxW, (W.shape[1], 1))).transpose()
        binaryW = W == maxMatW
        connMatW = np.dot(binaryW, binaryW.transpose())
        return connMatW
    #this　function is designed for calculateing the value in consensus matrix H
    def calcConnectivityH(self, H):
        maxH = H.to_numpy().max(axis=0)
        maxH[maxH == 0] = 1
        # argmaxH = H.to_numpy().argmax(axis=0)
        maxMatH = np.tile(maxH, (H.shape[0], 1))
        binaryH = H == maxMatH
        connMatH = np.dot(binaryH.transpose(), binaryH)
        return connMatH
    #this　function is designed for constructing the consensus matrix   
    def addConnectivityMatrixtoConsensusMatrix(self, connW, connH1, connH2, connH3, connH4, connH5, connH6, connPanH):
        self.cmW += connW
        self.cmH1 += connH1
        self.cmH2 += connH2
        self.cmH3 += connH3
        self.cmH4 += connH4
        self.cmH5 += connH5
        self.cmH6 += connH6
        self.cmPanH += connPanH
        self.cmNum += 1
        
    def finalizeConsensusMatrix(self, ndigits):
        self.cmW /= self.cmNum
        self.cmH1 /= self.cmNum
        self.cmH2 /= self.cmNum
        self.cmH3 /= self.cmNum
        self.cmH4 /= self.cmNum
        self.cmH5 /= self.cmNum
        self.cmH6 /= self.cmNum
        self.cmPanH /= self.cmNum
        np.fill_diagonal(self.cmW.to_numpy(), 1)
        np.fill_diagonal(self.cmH1.to_numpy(), 1)
        np.fill_diagonal(self.cmH2.to_numpy(), 1)
        np.fill_diagonal(self.cmH3.to_numpy(), 1)
        np.fill_diagonal(self.cmH4.to_numpy(), 1)
        np.fill_diagonal(self.cmH5.to_numpy(), 1)
        np.fill_diagonal(self.cmH6.to_numpy(), 1)
        np.fill_diagonal(self.cmPanH.to_numpy(), 1)
        self.cmW = self.cmW.round(ndigits)
        self.cmH1 = self.cmH1.round(ndigits)
        self.cmH2 = self.cmH2.round(ndigits)                
        self.cmH3 = self.cmH3.round(ndigits)
        self.cmH4 = self.cmH4.round(ndigits)
        self.cmH5 = self.cmH5.round(ndigits)                
        self.cmH6 = self.cmH6.round(ndigits)
    #this　function is designed for reordering, visualizing the consensus matrix and calculating the cophenetic correlation coefficient                    
    def reorderConsensusMatrix(self, M):
        Y = 1 - M
        Z = linkage(squareform(Y), method='average')
        #remove fastcluster.linkage
        Y1 = pdist(Y)
        ivl = leaves_list(Z)
        [c,D] = cophenet(Z,Y1)
        ivl = ivl[::-1]
        print(c)
        reorderM = pd.DataFrame(M.to_numpy()[:, ivl][ivl, :], index = M.columns[ivl], columns = M.columns[ivl])
        return reorderM, c
        