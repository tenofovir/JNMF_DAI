'''
Created on 2022/05/11

@author: YUTONG DAI
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


class SimulatedBinaryMissingData(object):
    '''
    classdocs
    generating the simulated data
    '''

    def makeSimulatedBinaryMissingData(self):
        '''
        setting four modules for compound data
                three modules for expression data
                three modules for mutation data
        '''
        w = np.ones((1,10))
        h1 = np.ones((1,30))
        h2 = np.ones((1,40))
        h3 = np.ones((1,50))
        
        W = np.zeros((45,4))    # 5
        H1 = np.zeros((4,130))  # 10
        H2 = np.zeros((4,170))  # 10
        H3 = np.zeros((4,215))  # 15
        
        W[0:10,0]=w
        W[10:20,1]=w
        W[20:30,2]=w
        W[30:40,3]=w
        
        H1[0,0:30]=h1
        H1[1,30:60]=h1
        H1[2,60:90]=h1        
        H1[3,90:120]=h1

        H2[0,0:40]=h2
        H2[1,40:80]=h2
        H2[2,80:120]=h2        
        #H2[3,120:160]=h2
        
        H3[0,0:50]=h3
        H3[1,50:100]=h3
        #H3[2,100:150]=h3        
        H3[3,150:200]=h3

        # original matrix
        X1=np.dot(W, H1)
        X2=np.dot(W, H2) 
        X3=np.dot(W, H3)

        # add noise
        alpha = 0.4
        XX1 = X1 + alpha * np.random.randn(45, 130)
        XX2 = X2 + alpha * np.random.randn(45, 170)     
        #XX3 = X3 + beta + alpha * np.random.rand(45, 215)     
        # genetic mutation matrix was represented in a binary format
        # generating the binary format noise
        noiseR3 = np.random.rand(45, 215)
        binaryR3 = noiseR3 > 0.95
        XX3 = (np.logical_xor(X3, binaryR3)).astype(float)
        # operate the exclusive OR operation the same = 0 otherwise = 1

        '''
        the noise was randomly generated
        1.remove the value which bigger than 1
        2.remove the negative value
        '''
        XX1[XX1>1] = 2-XX1[XX1>1]
        XX2[XX2>1] = 2-XX2[XX2>1]
        XX3[XX3>1] = 2-XX3[XX3>1]

        XX1[XX1<0] = -XX1[XX1<0]
        XX2[XX2<0] = -XX2[XX2<0]
        XX3[XX3<0] = -XX3[XX3<0]
        # remove the original negative value which bigger than 1
        XX1[XX1>1] = 2-XX1[XX1>1]
        XX2[XX2>1] = 2-XX2[XX2>1]
        XX3[XX3>1] = 2-XX3[XX3>1]
        # ensure that every value in matrix is positive
        XX1[XX1<0] = -XX1[XX1<0]
        XX2[XX2<0] = -XX2[XX2<0]
        XX3[XX3<0] = -XX3[XX3<0]
        # check the biggest value is 1
        XX1[XX1>1] = 1
        XX2[XX2>1] = 1
        XX3[XX3>1] = 1
        
        # adding missing values / arrays(nan) 10%rate
        binaryM1 = np.random.rand(45, 130) > 0.90
        XX1[binaryM1] = np.NaN
        binaryA2 = np.random.rand(45) > 0.90
        XX2[binaryA2] = np.NaN
        binaryA3 = np.random.rand(45) > 0.90
        XX3[binaryA3] = np.NaN       
               
        # random permutation
        p = np.random.permutation(45)
        p1 = np.random.permutation(130)
        p2 = np.random.permutation(170)
        p3 = np.random.permutation(215)

        # randomly permute by rows,early
        RX1 = XX1[p,:]
        RX2 = XX2[p,:]
        RX3 = XX3[p,:]

        # randomly permute by columns,late
        RXX1 = RX1[:,p1]
        RXX2 = RX2[:,p2]
        RXX3 = RX3[:,p3]

        # obtaining the random matrix
        simX1 = pd.DataFrame(RXX1)
        simX2 = pd.DataFrame(RXX2)        
        simX3 = pd.DataFrame(RXX3)
        
        return simX1, simX2, simX3
        
    def plotMatrix(self, M):  
        imshow(M, cmap='Blues', interpolation="nearest")
        plt.colorbar()

    def __init__(self):
        '''
        Constructor
        '''
        self.makeSimulatedBinaryMissingData()

'''
p = SimulatedBinaryMissingData()
A1,A2,A3 = p.makeSimulatedBinaryMissingData()
p.plotMatrix(A3)
plt.show()
'''