'''
Created on 2022/05/11

@author: YUTONG DAI
'''

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from matplotlib.pyplot import savefig, imshow, set_cmap
from classJointNMF_mask import *
from classConsensusMatrix import *
from classSaveData_of_TriNMF import *

class Run_TriNMF(object):
    '''
    classdocs
    '''

    def __init__(self, X1, X2, X3, maskX1, maskX2, maskX3, K, maxiter, nloop):
        '''
        Constructor
        '''
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.maskX1 = maskX1
        self.maskX2 = maskX2
        self.maskX3 = maskX3
        self.K = K
        self.maxiter = maxiter
        self.nloop = nloop
    
    def runTriNMF_singlerun(self, K, iloop):
        self.K = K
        
        """
        Run Joint NMF
        """
        jnmf = JointNMF_mask(self.X1, self.X2, self.X3, self.maskX1, self.maskX2, self.maskX3, self.K, self.maxiter)
        jnmf.check_nonnegativity()
        jnmf.check_samplesize()
        jnmf.initialize_W_H()
        #jnmf.update_euclidean_multiplicative()
        jnmf.wrapper_calc_euclidean_multiplicative_update()
        jnmf.print_distance_of_HW_to_X(iloop)
        jnmf.set_PanH()
        return jnmf
                
    def runTriNMF_multirun(self, K, nloop):
        self.K = K
        self.nloop = nloop
        
        """
        Run Joint NMF
        Calculate consensus matrix
        """
        cmatrix = ConsensusMatrix(self.X1, self.X2, self.X3)
        sdata = SaveDataWH(self.X1, self.X2, self.X3, self.K)
        for i in range(self.nloop):
            jnmf = self.runTriNMF_singlerun(self.K, i)   
            connW = cmatrix.calcConnectivityW(jnmf.W)
            connH1 = cmatrix.calcConnectivityH(jnmf.H1)
            connH2 = cmatrix.calcConnectivityH(jnmf.H2)
            connH3 = cmatrix.calcConnectivityH(jnmf.H3)
            connPanH = cmatrix.calcConnectivityH(jnmf.PanH)
            cmatrix.addConnectivityMatrixtoConsensusMatrix(connW, connH1, connH2, connH3, connPanH)
            sdata.save_W_H(jnmf.W, jnmf.H1, jnmf.H2, jnmf.H3, K, i)
            
        cmatrix.finalizeConsensusMatrix()

        """
        Save output files: W, H1, H2, and H3
        """
        return cmatrix, sdata




