'''
Created on 2022/05/11

@author: Yutong Dai

This part is the updated formula part of the algorithm, and the specific formula is consistent as derived in the main text

'''
import numpy as np
import pandas as pd

class HexaNMF_mask(object):
    '''
    Joint NMF
    '''


    def __init__(self, X1, X2, X3, X4, X5, X6, maskX1, maskX2, maskX3, maskX4, maskX5, maskX6, rank, maxiter):
        '''
        Constructor
        '''
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.X4 = X4
        self.X5 = X5
        self.X6 = X6
        self.maskX1 = maskX1
        self.maskX2 = maskX2
        self.maskX3 = maskX3
        self.maskX4 = maskX4
        self.maskX5 = maskX5
        self.maskX6 = maskX6
        self.rank = rank
        self.maxiter = maxiter
        self.report = pd.DataFrame(columns = ["K", "iter", "difference of WH - WH_pre", "Euclidean distance of X - WH", "MASKED Euclidean distance of X - WH", "Error: avg' |X - WH|/X"])
    
    #this　function is designed for Ensuring non-negativity    
    def check_nonnegativity(self):
        if(self.X1.min().min() < 0 or self.X2.min().min() < 0 or self.X3.min().min() < 0 or self.X4.min().min() < 0 or self.X5.min().min() < 0 or self.X6.min().min() < 0):
            raise Exception('non negativity')
    #this　function is designed for checking the input size of input matrix
    def check_samplesize(self):
        if(self.X1.shape[0] != self.X2.shape[0] or self.X1.shape[0] != self.X3.shape[0] or self.X1.shape[0] != self.X4.shape[0] or self.X1.shape[0] != self.X5.shape[0] or self.X1.shape[0] != self.X6.shape[0]):
            print(self.X1.shape[0])
            print(self.X2.shape[0])
            print(self.X3.shape[0])
            print(self.X4.shape[0])
            print(self.X5.shape[0])
            print(self.X6.shape[0])
            raise Exception('sample size')

    #this　function is designed for randomly initializing the matrix       
    def initialize_W_H(self):
        self.W = pd.DataFrame(np.random.rand(self.X1.shape[0], self.rank), index = self.X1.index, columns = map(str, range(1, self.rank+1)))
        self.H1 = pd.DataFrame(np.random.rand(self.rank, self.X1.shape[1]), index = map(str, range(1, self.rank+1)), columns = self.X1.columns)
        self.H2 = pd.DataFrame(np.random.rand(self.rank, self.X2.shape[1]), index = map(str, range(1, self.rank+1)), columns = self.X2.columns)
        self.H3 = pd.DataFrame(np.random.rand(self.rank, self.X3.shape[1]), index = map(str, range(1, self.rank+1)), columns = self.X3.columns)
        self.H4 = pd.DataFrame(np.random.rand(self.rank, self.X4.shape[1]), index = map(str, range(1, self.rank+1)), columns = self.X4.columns)
        self.H5 = pd.DataFrame(np.random.rand(self.rank, self.X5.shape[1]), index = map(str, range(1, self.rank+1)), columns = self.X5.columns)
        self.H6 = pd.DataFrame(np.random.rand(self.rank, self.X6.shape[1]), index = map(str, range(1, self.rank+1)), columns = self.X6.columns)
        self.X1r_pre = np.dot(self.W, self.H1)
        self.X2r_pre = np.dot(self.W, self.H2)
        self.X3r_pre = np.dot(self.W, self.H3)
        self.X4r_pre = np.dot(self.W, self.H4)
        self.X5r_pre = np.dot(self.W, self.H5)
        self.X6r_pre = np.dot(self.W, self.H6)
        self.eps = np.finfo(self.W.to_numpy().dtype).eps

    #this　function is designed for implementation of the update formula (as demonstrated in the main text)
    def calc_euclidean_multiplicative_update(self):
        self.H1 = np.multiply(self.H1, np.divide(np.dot(self.W.T, np.multiply(self.maskX1, self.X1)), np.dot(self.W.T, np.multiply(self.maskX1, np.dot(self.W, self.H1)+self.eps))))
        self.H2 = np.multiply(self.H2, np.divide(np.dot(self.W.T, np.multiply(self.maskX2, self.X2)), np.dot(self.W.T, np.multiply(self.maskX2, np.dot(self.W, self.H2)+self.eps))))
        self.H3 = np.multiply(self.H3, np.divide(np.dot(self.W.T, np.multiply(self.maskX3, self.X3)), np.dot(self.W.T, np.multiply(self.maskX3, np.dot(self.W, self.H3)+self.eps))))
        self.H4 = np.multiply(self.H4, np.divide(np.dot(self.W.T, np.multiply(self.maskX4, self.X4)), np.dot(self.W.T, np.multiply(self.maskX4, np.dot(self.W, self.H4)+self.eps))))
        self.H5 = np.multiply(self.H5, np.divide(np.dot(self.W.T, np.multiply(self.maskX5, self.X5)), np.dot(self.W.T, np.multiply(self.maskX5, np.dot(self.W, self.H5)+self.eps))))
        self.H6 = np.multiply(self.H6, np.divide(np.dot(self.W.T, np.multiply(self.maskX6, self.X6)), np.dot(self.W.T, np.multiply(self.maskX6, np.dot(self.W, self.H6)+self.eps))))
        maskPanX = np.c_[self.maskX1, self.maskX2, self.maskX3, self.maskX4, self.maskX5, self.maskX6]
        PanX = np.c_[self.X1, self.X2, self.X3, self.X4, self.X5, self.X6]
        PanH = np.c_[self.H1, self.H2, self.H3, self.H4, self.H5, self.H6]
        self.W = np.multiply(self.W, np.divide(np.dot(np.multiply(maskPanX, PanX), np.transpose(PanH)), (np.dot(np.multiply(maskPanX, np.dot(self.W, PanH)), np.transpose(PanH))+self.eps)))
    #This function is designed for updating the W and H matrices according to a set iteration count
    def wrapper_calc_euclidean_multiplicative_update(self):
        for run in range(self.maxiter):
            self.calc_euclidean_multiplicative_update()
            self.calc_distance_of_HW_to_X()
            #self.print_distance_report_of_HW_to_X(run)
            self.get_distance_report_of_HW_to_X(run)
        return self.report
    #This function is designed for calculateing the euclidean distance between two matrices
    def calc_distance_of_HW_to_X(self):
        self.X1r = np.dot(self.W, self.H1)
        self.X2r = np.dot(self.W, self.H2)
        self.X3r = np.dot(self.W, self.H3)
        self.X4r = np.dot(self.W, self.H4)
        self.X5r = np.dot(self.W, self.H5)
        self.X6r = np.dot(self.W, self.H6)
        self.diff = np.sum(np.sum(np.abs(self.X1r_pre-self.X1r))) + np.sum(np.sum(np.abs(self.X2r_pre-self.X2r))) + np.sum(np.sum(np.abs(self.X3r_pre-self.X3r))) + np.sum(np.sum(np.abs(self.X4r_pre-self.X4r))) + np.sum(np.sum(np.abs(self.X5r_pre-self.X5r))) + np.sum(np.sum(np.abs(self.X6r_pre-self.X6r)))
        self.X1r_pre = self.X1r
        self.X2r_pre = self.X2r
        self.X3r_pre = self.X3r
        self.X4r_pre = self.X4r
        self.X5r_pre = self.X5r
        self.X6r_pre = self.X6r
        self.eucl_dist1 = self.calc_euclidean_dist(self.X1, self.X1r)
        self.eucl_dist2 = self.calc_euclidean_dist(self.X2, self.X2r)
        self.eucl_dist3 = self.calc_euclidean_dist(self.X3, self.X3r)
        self.eucl_dist4 = self.calc_euclidean_dist(self.X4, self.X4r)
        self.eucl_dist5 = self.calc_euclidean_dist(self.X5, self.X5r)
        self.eucl_dist6 = self.calc_euclidean_dist(self.X6, self.X6r)
        self.eucl_dist = self.eucl_dist1 + self.eucl_dist2 + self.eucl_dist3 + self.eucl_dist4 + self.eucl_dist5 + self.eucl_dist6
        self.masked_eucl_dist1 = self.calc_masked_euclidean_dist(self.X1, self.X1r, self.maskX1)
        self.masked_eucl_dist2 = self.calc_masked_euclidean_dist(self.X2, self.X2r, self.maskX2)
        self.masked_eucl_dist3 = self.calc_masked_euclidean_dist(self.X3, self.X3r, self.maskX3)
        self.masked_eucl_dist4 = self.calc_masked_euclidean_dist(self.X4, self.X4r, self.maskX4)
        self.masked_eucl_dist5 = self.calc_masked_euclidean_dist(self.X5, self.X5r, self.maskX5)
        self.masked_eucl_dist6 = self.calc_masked_euclidean_dist(self.X6, self.X6r, self.maskX6)
        self.masked_eucl_dist = self.masked_eucl_dist1 + self.masked_eucl_dist2 + self.masked_eucl_dist3 + self.masked_eucl_dist4 + self.masked_eucl_dist5 + self.masked_eucl_dist6
        self.error1 = np.mean(np.mean(np.abs(self.X1-self.X1r)))/np.mean(np.mean(self.X1))
        self.error2 = np.mean(np.mean(np.abs(self.X2-self.X2r)))/np.mean(np.mean(self.X2))
        self.error3 = np.mean(np.mean(np.abs(self.X3-self.X3r)))/np.mean(np.mean(self.X3))
        self.error4 = np.mean(np.mean(np.abs(self.X4-self.X4r)))/np.mean(np.mean(self.X4))
        self.error5 = np.mean(np.mean(np.abs(self.X5-self.X5r)))/np.mean(np.mean(self.X5))
        self.error6 = np.mean(np.mean(np.abs(self.X6-self.X6r)))/np.mean(np.mean(self.X6))
        self.error = self.error1 + self.error2 + self.error3 + self.error4 + self.error5 + self.error6
        
    #This function is designed for　printing the euclidean distance    
    def print_distance_report_of_HW_to_X(self, text):
        print("[%s] diff = %f, eucl_dist = %f, MASKED_eucl_dist = %f, error = %f" % (text, self.diff, self.eucl_dist, self.masked_eucl_dist, self.error))
    #This function is designed for　adding the euclidean distance into report 
    def get_distance_report_of_HW_to_X(self, text):
        newreport = pd.DataFrame([[self.rank, text, self.diff, self.eucl_dist, self.masked_eucl_dist, self.error]], columns = ["K", "iter", "difference of WH - WH_pre", "Euclidean distance of X - WH", "MASKED Euclidean distance of X - WH", "Error: avg' |X - WH|/X"])
        self.report = self.report.append(newreport)       
        return self.report
        
    def calc_euclidean_dist(self, X, Y):
        dist = np.sum(np.sum(np.power(X-Y, 2)))
        return dist

    def calc_masked_euclidean_dist(self, X, Y, maskX):
        dist = np.sum(np.sum(np.power(np.multiply(maskX, X-Y), 2)))
        return dist
    
    def set_PanH(self):
        self.PanH = pd.concat([self.H1, self.H2, self.H3, self.H4, self.H5, self.H6], axis=1)
        columnsPanH = ["X1_" + str(x) for x in self.H1.columns] + ["X2_" + str(x) for x in self.H2.columns] + ["X3_" + str(x) for x in self.H3.columns] + ["X4_" + str(x) for x in self.H4.columns] + ["X5_" + str(x) for x in self.H5.columns] + ["X6_" + str(x) for x in self.H6.columns]
        self.PanH.columns = columnsPanH  
            
#    def run(self):
        