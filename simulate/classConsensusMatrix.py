'''
Created on 2022/05/11

@author: YUTONG DAI
'''
import numpy as np
import pandas as pd
import numpy.matlib
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

class ConsensusMatrix(object):
    '''
    Connectivity matrix (single JNMF run)
    Consensus matrix (multiple JNMF run)
    Reordring consensus matrix
    Cophenetic correlation

    testing how well the clustering result matches the original resemblances
    we use ConsensusMatrix and hierarchy clustering here(Cophenetic correlation)



    '''


    def __init__(self, X1, X2, X3):
        '''
        Constructor
        '''
        self.cmW = pd.DataFrame(np.zeros((X1.shape[0], X1.shape[0])), index = X1.index, columns = X1.index)
        self.cmH1 = pd.DataFrame(np.zeros((X1.shape[1], X1.shape[1])), index = X1.columns, columns = X1.columns)
        self.cmH2 = pd.DataFrame(np.zeros((X2.shape[1], X2.shape[1])), index = X2.columns, columns = X2.columns)
        self.cmH3 = pd.DataFrame(np.zeros((X3.shape[1], X3.shape[1])), index = X3.columns, columns = X3.columns)
        sizePanH = X1.shape[1] + X2.shape[1] + X3.shape[1]
        columnsPanH = ["X1_" + str(x) for x in X1.columns] + ["X2_" + str(x) for x in X2.columns] + ["X3_" + str(x) for x in X3.columns]
        self.cmPanH = pd.DataFrame(np.zeros((sizePanH, sizePanH)), index = columnsPanH, columns = columnsPanH)
        self.cmNum = 0
    # ???
    def calcConnectivityW(self, W):
        # by rows
        maxW = W.to_numpy().max(axis=1)
        # checking none 0 in maxW
        maxW[maxW == 0] = 1

        # change the argmaxW = W.to_numpy.argmax(axis=1) test1:remove
        # constructing the matrix by repeating the maxW by (columns of W,1), covert the vector into a square matrix
        maxMatW = (np.tile(maxW, (W.shape[1], 1))).transpose()
        # checking the position of the biggest values
        binaryW = W == maxMatW
        # built symmetric matrix
        connMatW = np.dot(binaryW, binaryW.transpose())

        return connMatW
    
    def calcConnectivityH(self, H):
        # finding the biggest value by columns(axis = 0) when(axis =1) by rows: generating a vector
        maxH = H.to_numpy().max(axis=0)
        maxH[maxH == 0] = 1

        # change the argmaxH = H.to_numpy.argmax(axis=0) test1:remove
        maxMatH = np.tile(maxH, (H.shape[0], 1))
        binaryH = H == maxMatH
        connMatH = np.dot(binaryH.transpose(), binaryH)
        return connMatH
        
    def addConnectivityMatrixtoConsensusMatrix(self, connW, connH1, connH2, connH3, connPanH):
        # operate the matrix addition
        # if the biggest value occur in different position, the true will be added to the binary matrix
        self.cmW += connW
        self.cmH1 += connH1
        self.cmH2 += connH2
        self.cmH3 += connH3
        self.cmPanH += connPanH
        self.cmNum += 1
        
    def finalizeConsensusMatrix(self):
        self.cmW /= self.cmNum
        self.cmH1 /= self.cmNum
        self.cmH2 /= self.cmNum
        self.cmH3 /= self.cmNum
        self.cmPanH /= self.cmNum
        np.fill_diagonal(self.cmW.to_numpy(), 1)
        np.fill_diagonal(self.cmH1.to_numpy(), 1)
        np.fill_diagonal(self.cmH2.to_numpy(), 1)
        np.fill_diagonal(self.cmH3.to_numpy(), 1)
        np.fill_diagonal(self.cmPanH.to_numpy(), 1)
        self.cmW = self.reorderConsensusMatrix(self.cmW)
        self.cmH1 = self.reorderConsensusMatrix(self.cmH1)
        self.cmH2 = self.reorderConsensusMatrix(self.cmH2)
        self.cmH3 = self.reorderConsensusMatrix(self.cmH3)
        self.cmPanH = self.reorderConsensusMatrix(self.cmPanH)

    # ???
    def reorderConsensusMatrix(self, M):
        '''
        operating Hierarchical Clustering
        using Cophenetic correlation is a measure of how well the clustering result matches the original resemblances
        in n iterations, if biggest value occur in different position, the distance information will be extracted by hierarchy cluster

        squareform():Converts a vector-form distance vector to a square-form distance matrix,and vice-versa
        here we input the square-form distance matrix, so the vector-form distance are outputted here

        linkage Perform hierarchical clustering.
        '''
        Y = 1 - M
        Z = linkage(squareform(Y), method='average')
        Y1 = pdist(Y)
        ivl = leaves_list(Z)
        [c, D] = cophenet(Z, Y1)
        # reverse the ivl
        ivl = ivl[::-1]
        print(c)
        reorderM = pd.DataFrame(M.to_numpy()[:, ivl][ivl, :], index = M.columns[ivl], columns = M.columns[ivl])
        return reorderM

    '''
    calculate cophenetic correlation coefficient(c)
    use the (from scipy.cluster.hierarchy import cophenet)
    in this case:
    
    cophenet(cluster result, condensed distance matrix(shape=1) )
    cluster result:Z = linkage()
    
    condensed distance matrix: use (from scipy.spatial.distance import pdist) 
    Condensed distance matrix: Y = pdist(Y = 1-M(here))original input matrix
    
    result:
    [c,D] = cophenet(Z,Y)
    c: the cophenetic correlation coefficient
    D: The cophentic correlation distance matrix
    '''

    # as_matrix to to_numpy