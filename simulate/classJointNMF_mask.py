'''
Created on 2022/05/11

@author: YUTONG DAI
'''
import numpy as np
import pandas as pd
from scipy import stats

class JointNMF_mask(object):
    '''
    Joint NMF
    '''


    def __init__(self, X1, X2, X3, maskX1, maskX2, maskX3, rank, maxiter):
        '''
        Constructor
        '''
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.maskX1 = maskX1
        self.maskX2 = maskX2
        self.maskX3 = maskX3
        self.rank = rank
        self.maxiter = maxiter
        # initiation

    def check_nonnegativity(self):
        if(self.X1.min().min() < 0 or self.X2.min().min() < 0 or self.X3.min().min() < 0):
            raise Exception('non negativity')
       #checking the data : whether every elements in matrix is non negativity
    
    def check_samplesize(self):
        if(self.X1.shape[0] != self.X2.shape[0] or self.X1.shape[0] != self.X3.shape[0]):
            raise Exception('sample size')
        #checking the data : 
     



    def initialize_W_H(self):
        self.W = pd.DataFrame(np.random.rand(self.X1.shape[0], self.rank), index = self.X1.index, columns = map(str, range(1, self.rank+1)))
        self.H1 = pd.DataFrame(np.random.rand(self.rank, self.X1.shape[1]), index = map(str, range(1, self.rank+1)), columns = self.X1.columns)
        self.H2 = pd.DataFrame(np.random.rand(self.rank, self.X2.shape[1]), index = map(str, range(1, self.rank+1)), columns = self.X2.columns)
        self.H3 = pd.DataFrame(np.random.rand(self.rank, self.X3.shape[1]), index = map(str, range(1, self.rank+1)), columns = self.X3.columns)
        self.X1r_pre = np.dot(self.W, self.H1)
        self.X2r_pre = np.dot(self.W, self.H2)
        self.X3r_pre = np.dot(self.W, self.H3)
        # ???
        self.eps = np.finfo(self.W.to_numpy().dtype).eps
    # initializing the W and H matrix rank: features we want extract


    def calc_euclidean_multiplicative_update(self):
        '''uppdate rule:
     1 H1 =  H1*(W.T(maskX1*X1))/(W.T(maskX1*(WH1)))
     2 H2 =  H2*(W.T(maskX2*X2))/(W.T(maskX2*(WH2)))
     3 H3 =  H3*(W.T(maskX3*X3))/(W.T(maskX3*(WH3)))
     4 W = W*((maskX1*X1)H1.T+(maskX2*X2)H2.T+(maskX3*X3)H3.T)/(maskX1*WH1H1.T+maskX2*WH2H2.T+maskX3*WH3H3.T))
     '''

        self.H1 = np.multiply(self.H1, np.divide(np.dot(self.W.T, np.multiply(self.maskX1, self.X1)), np.dot(self.W.T, np.multiply(self.maskX1, np.dot(self.W, self.H1)+self.eps))))
        self.H2 = np.multiply(self.H2, np.divide(np.dot(self.W.T, np.multiply(self.maskX2, self.X2)), np.dot(self.W.T, np.multiply(self.maskX2, np.dot(self.W, self.H2)+self.eps))))
        self.H3 = np.multiply(self.H3, np.divide(np.dot(self.W.T, np.multiply(self.maskX3, self.X3)), np.dot(self.W.T, np.multiply(self.maskX3, np.dot(self.W, self.H3)+self.eps))))
        self.W = np.multiply(self.W, np.divide(np.dot(np.multiply(np.c_[self.maskX1, self.maskX2, self.maskX3], np.c_[self.X1, self.X2, self.X3]), np.transpose(np.c_[self.H1, self.H2, self.H3])), (np.dot(np.multiply(np.c_[self.maskX1, self.maskX2, self.maskX3], np.dot(self.W, np.c_[self.H1, self.H2, self.H3])), np.transpose(np.c_[self.H1, self.H2, self.H3]))+self.eps)))
     



    def wrapper_calc_euclidean_multiplicative_update(self):
        # calculate update loop maxiter times
        for run in range(self.maxiter):
            self.calc_euclidean_multiplicative_update()
            self.calc_distance_of_HW_to_X()
            #self.print_distance_of_HW_to_X(run)       
    
    def calc_distance_of_HW_to_X(self):
        self.X1r = np.dot(self.W, self.H1)
        self.X2r = np.dot(self.W, self.H2)
        self.X3r = np.dot(self.W, self.H3)
        self.diff = np.sum(np.sum(np.abs(self.X1r_pre-self.X1r))) + np.sum(np.sum(np.abs(self.X2r_pre-self.X2r))) + np.sum(np.sum(np.abs(self.X3r_pre-self.X3r)))
        self.X1r_pre = self.X1r
        self.X2r_pre = self.X2r
        self.X3r_pre = self.X3r
        self.eucl_dist1 = self.calc_euclidean_dist(self.X1, self.X1r)
        self.eucl_dist2 = self.calc_euclidean_dist(self.X2, self.X2r)
        self.eucl_dist3 = self.calc_euclidean_dist(self.X3, self.X3r)
        self.eucl_dist = self.eucl_dist1 + self.eucl_dist2 + self.eucl_dist3
        self.error1 = np.mean(np.mean(np.abs(self.X1-self.X1r)))/np.mean(np.mean(self.X1))
        # compute the means by columns / using Percentage Error Formula (x1(origin matrix)-x1r(after iteration))/X1
        self.error2 = np.mean(np.mean(np.abs(self.X2-self.X2r)))/np.mean(np.mean(self.X2))
        self.error3 = np.mean(np.mean(np.abs(self.X3-self.X3r)))/np.mean(np.mean(self.X3))
        self.error = self.error1 + self.error2 + self.error3
        
        
    def print_distance_of_HW_to_X(self, text):
        print("[%s] diff = %f, eucl_dist = %f, error = %f" % (text, self.diff, self.eucl_dist, self.error))
               
         
    def calc_euclidean_dist(self, X, Y):
        dist = np.sum(np.sum(np.power(X-Y, 2)))
        return dist

    # ???
    def set_PanH(self):
        # Concatenate matrix by columns(axis = 1) H1H2H3
        self.PanH = pd.concat([self.H1, self.H2, self.H3], axis=1)
        # Assign the index of the columns of Panh with the index of the columns of each H(123)
        columnsPanH = ["X1_" + str(x) for x in self.H1.columns] + ["X2_" + str(x) for x in self.H2.columns] + ["X3_" + str(x) for x in self.H3.columns]
        self.PanH.columns = columnsPanH  
        
    def reorder_cluster_of_HWX(self):
        '''
        1,finding the indexes of biggest values in W and H
        2,reorder the matrix(W,H,X) based on the indexes
        3. constructing the clustered matrix with a diagonal form
        '''



        # Return index of the maximum element by rows(1) columns(0)
        # (0) right:the index of columns left: the index of rows (1) vice versa
        clusterWid = self.W.idxmax(1)
        clusterH1id = self.H1.idxmax(0)
        clusterH2id = self.H2.idxmax(0)
        clusterH3id = self.H3.idxmax(0)
        
        # filtering by entropy
        #threshold = stats.entropy(np.zeros(self.rank)+self.eps) * 0.95
        #clusterWid[stats.entropy(self.W.T+self.eps) > threshold] = str(self.rank + 1)      
        #clusterH1id[stats.entropy(self.H1+self.eps) > threshold] = str(self.rank + 1)    
        #clusterH2id[stats.entropy(self.H2+self.eps) > threshold] = str(self.rank + 1)    
        #clusterH3id[stats.entropy(self.H3+self.eps) > threshold] = str(self.rank + 1)           

        # filtering by max  removing the outliers
        # Calculate the average of the maximum value by rows
        valmax = self.W.max(1).mean()
        # Return standard deviation of matrix
        valstddev = self.W.max(1).std()
        # Check whether there are outliers.
        # Generally, data with more than two standard deviations is considered an outlier
        # The data is required to follow a normal distribution.
        clusterWid[self.W.max(1) < valmax - 2 * valstddev] = str(self.rank + 1)
        # Calculate the average of the maximum value by columns
        valmax = self.H1.max(0).mean()
        valstddev = self.H1.max(0).std()
        clusterH1id[self.H1.max(0) < valmax - 2 * valstddev] = str(self.rank + 1)
        valmax = self.H2.max(0).mean()
        valstddev = self.H2.max(0).std()
        clusterH2id[self.H2.max(0) < valmax - 1 * valstddev] = str(self.rank + 1)
        valmax = self.H3.max(0).mean()
        valstddev = self.H3.max(0).std()
        clusterH3id[self.H3.max(0) < valmax - 1 * valstddev] = str(self.rank + 1)

        # return the index(position) of sorting result from small to big
        orderWid = np.argsort(clusterWid)
        orderH1id = np.argsort(clusterH1id)
        orderH2id = np.argsort(clusterH2id)
        orderH3id = np.argsort(clusterH3id)
        # iloc  is based on location index
        # loc is based on  label index
        self.rW = self.W.iloc[orderWid,:]
        self.rH1 = self.H1.iloc[:,orderH1id]
        self.rH2 = self.H2.iloc[:,orderH2id]
        self.rH3 = self.H3.iloc[:,orderH3id]
        self.rX1 = self.X1.iloc[orderWid,orderH1id]
        self.rX2 = self.X2.iloc[orderWid,orderH2id]
        self.rX3 = self.X3.iloc[orderWid,orderH3id]
        self.rWH1 = pd.DataFrame(np.dot(self.rW, self.rH1), index = self.X1.index, columns = self.X1.columns)
        self.rWH2 = pd.DataFrame(np.dot(self.rW, self.rH2), index = self.X2.index, columns = self.X2.columns)
        self.rWH3 = pd.DataFrame(np.dot(self.rW, self.rH3), index = self.X3.index, columns = self.X3.columns)
        self.orderWid = orderWid
        self.orderH1id = orderH1id         
        self.orderH2id = orderH2id  
        self.orderH3id = orderH3id
#         .ix into .iloc
#    def run(self):
#    as_matrix to to_numpy
