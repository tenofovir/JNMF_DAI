'''
Created on 2022/05/11

@author: Yutong Dai

This section is designed to store the output matrices
'''
import numpy as np
import pandas as pd

class SaveHexaDataWH(object):
    '''
    Joint NMF
    '''

    def __init__(self, X1, X2, X3, X4, X5, X6, rank):
        '''
        Constructor
        '''
        self.rank = rank
        self.W = pd.DataFrame()
        self.H1 = pd.DataFrame(columns = X1.columns)
        self.H2 = pd.DataFrame(columns = X2.columns)
        self.H3 = pd.DataFrame(columns = X3.columns)
        self.H4 = pd.DataFrame(columns = X4.columns)
        self.H5 = pd.DataFrame(columns = X5.columns)
        self.H6 = pd.DataFrame(columns = X6.columns)
    # Saving the W and H matrices 
    def save_W_H(self, W, H1, H2, H3, H4, H5, H6, rank, trial):
        self.W = self.W.append(self.make_M(W, rank, trial))
        self.H1 = self.H1.append(self.make_M(H1, rank, trial))
        self.H2 = self.H2.append(self.make_M(H2, rank, trial))
        self.H3 = self.H3.append(self.make_M(H3, rank, trial))
        self.H4 = self.H4.append(self.make_M(H4, rank, trial))
        self.H5 = self.H5.append(self.make_M(H5, rank, trial))
        self.H6 = self.H6.append(self.make_M(H6, rank, trial))
            
    def make_M(self, M, rank, trial):
        M['rank'] = rank
        M['trial'] = trial
        return M

class SaveBestHexaDataWH(object):
    '''
    Save Best Matrix from Joint NMF results
    '''

    def __init__(self, X1, X2, X3, X4, X5, X6, rank):
        '''
        Constructor
        '''
        self.rank = rank
        self.W = pd.DataFrame()
        self.H1 = pd.DataFrame(columns = X1.columns)
        self.H2 = pd.DataFrame(columns = X2.columns)
        self.H3 = pd.DataFrame(columns = X3.columns)
        self.H4 = pd.DataFrame(columns = X4.columns)
        self.H5 = pd.DataFrame(columns = X5.columns)
        self.H6 = pd.DataFrame(columns = X6.columns)
        self.bestscore = float("inf")
                
    def save_W_H(self, W, H1, H2, H3, H4, H5, H6, rank, trial):
        self.W = self.make_M(W, rank, trial)
        self.H1 = self.make_M(H1, rank, trial)
        self.H2 = self.make_M(H2, rank, trial)
        self.H3 = self.make_M(H3, rank, trial)
        self.H4 = self.make_M(H4, rank, trial)
        self.H5 = self.make_M(H5, rank, trial)
        self.H6 = self.make_M(H6, rank, trial)
    
    def make_M(self, M, rank, trial):
        M['rank'] = rank
        M['trial'] = trial
        return M
    
#    def run(self):
        