'''
Created on 2022/05/11

@author: YUTONG DAI


'''


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy import stats
from matplotlib.pyplot import savefig, imshow, set_cmap
from classJointNMF_mask  import *
from classConsensusMatrix import *
from classSimulatedData_binary_missing import *
from classRun_TriNMF_CCLE_mask import *

def main():
    
    """
    Generate simulated data files: X1, X2, and X3
    X: The step that check whether NAns are in the matrix
    maskX is the weight matrix.

    """
    sim = SimulatedBinaryMissingData()
    X1ori, X2ori, X3ori = sim.makeSimulatedBinaryMissingData()
    
    X1 = X1ori[X1ori.notnull().any(axis=1)] # checking the Nan value
    # generating the binary matrix(weight matrix): removing the interference of missing value
    maskX1 = 1 - X1.isnull()
    # removing the missing value
    X1 = X1.fillna(0)
    
    X2 = X2ori[X1ori.notnull().any(axis=1)]
    maskX2 = 1 - X2.isnull()
    X2 = X2.fillna(0)
    
    X3 = X3ori[X1ori.notnull().any(axis=1)]
    maskX3 = 1 - X3.isnull()
    X3 = X3.fillna(0)
     
    """
    Set parameters
    """
    nloop = 10
    maxiter = 1100
    K = 4



    
    """
    Set output directory
    """
    # savedir = '/home/yutong/figure%d_n%d_iter%d' % (K, nloop, maxiter)
    savedir = '/Users/yutongdai/Desktop/paper/figure%d_n%d_iter%d' % (K, nloop, maxiter)
    if not os.path.exists(savedir):
        os.mkdir(savedir)
        
    
    """
    Save generated input files: X1, X2, and X3
    """
    X1ori.to_csv(savedir + '/jnmf_input_X1_CCLE_k%d.csv' % K)
    X2ori.to_csv(savedir + '/jnmf_input_X2_CCLE_k%d.csv' % K)
    X3ori.to_csv(savedir + '/jnmf_input_X3_CCLE_k%d.csv' % K)
    
    cmap = cm.Purples
    # Set the color for masked values.
    cmap.set_bad('gray', 1)

    imshow(X1ori, cmap=cmap, interpolation="nearest", aspect='auto')
    savefig(savedir + '/jnmf_input_X1_CCLE_k%d.png' % K)
    plt.colorbar()
    plt.close()
    
    imshow(X2ori, cmap=cmap, interpolation="nearest", aspect='auto')
    savefig(savedir + '/jnmf_input_X2_CCLE_k%d.png' % K)
    plt.colorbar()
    plt.close()
    
    imshow(X3ori, cmap=cmap, interpolation="nearest", aspect='auto')
    savefig(savedir + '/jnmf_input_X3_CCLE_k%d.png' % K)
    # adding colorbars not attached to a previously drawn picture
    plt.colorbar()
    plt.close()
    
    """
    Run Joint NMF
    Calculate consensus matrix
    """
    nmf = Run_TriNMF(X1, X2, X3, maskX1, maskX2, maskX3, K, maxiter, nloop)
    jnmf = nmf.runTriNMF_singlerun(K, 1)
    cmatrix, sdata = nmf.runTriNMF_multirun(K, nloop)
    
    """
    Save output files: W, H1, H2, and H3
    """
    sdata.W.to_csv(savedir + '/jnmf_dataW_CCLE_k%d.csv' % K)
    sdata.H1.to_csv(savedir + '/jnmf_dataH1_CCLE_k%d.csv' % K)
    sdata.H2.to_csv(savedir + '/jnmf_dataH2_CCLE_k%d.csv' % K)
    sdata.H3.to_csv(savedir + '/jnmf_dataH3_CCLE_k%d.csv' % K)
    
    """
    Plot consensus matrix.
    """    
    imshow(cmatrix.cmW, cmap='Purples', interpolation="nearest")
    plt.colorbar()
    savefig(savedir + '/jnmf_CCLE_ConsensusMatrixW_k%d.png' % K)
    plt.close()
    
    imshow(cmatrix.cmPanH, cmap='Purples', interpolation="nearest")
    plt.colorbar()
    savefig(savedir + '/jnmf_CCLE_ConsensusMatrixPanH_k%d.png' % K)
    plt.close()
    
    """
    Save output files: cmW, cmPanH
    """
    cmatrix.cmW.to_csv(savedir + '/jnmf_CCLE_ConsensusMatrixW_k%d.csv' % K)
    cmatrix.cmPanH.to_csv(savedir + '/jnmf_CCLE_ConsensusMatrixPanH_k%d.csv' % K)
    
    """
    Plot output files: W, H1, H2, H3, cmW, and  cmPanH
    """
    jnmf.reorder_cluster_of_HWX()
    rX1ori = X1ori.iloc[jnmf.orderWid,jnmf.orderH1id]
    rX2ori = X2ori.iloc[jnmf.orderWid,jnmf.orderH2id]
    rX3ori = X3ori.iloc[jnmf.orderWid,jnmf.orderH3id]
    # ix into iloc
    jnmf.rW.to_csv(savedir + '/jnmf_single_rW_CCLE_k%d.csv' % K)
    jnmf.rH1.to_csv(savedir + '/jnmf_single_rH1_CCLE_k%d.csv' % K)
    jnmf.rH2.to_csv(savedir + '/jnmf_single_rH2_CCLE_k%d.csv' % K)
    jnmf.rH3.to_csv(savedir + '/jnmf_single_rH3_CCLE_k%d.csv' % K)
    jnmf.rWH1.to_csv(savedir + '/jnmf_single_rWH1_CCLE_k%d.csv' % K)
    jnmf.rWH2.to_csv(savedir + '/jnmf_single_rWH2_CCLE_k%d.csv' % K)
    jnmf.rWH3.to_csv(savedir + '/jnmf_single_rWH3_CCLE_k%d.csv' % K)
    rX1ori.to_csv(savedir + '/jnmf_single_rX1_CCLE_k%d.csv' % K)
    rX2ori.to_csv(savedir + '/jnmf_single_rX2_CCLE_k%d.csv' % K)
    rX3ori.to_csv(savedir + '/jnmf_single_rX3_CCLE_k%d.csv' % K)
    
    cmap = cm.Purples
    cmap.set_bad('gray', 1)
    # showing the figures
    imshow(jnmf.rW, cmap=cmap, interpolation="nearest", aspect='auto')
    savefig(savedir + '/jnmf_single_rW_CCLE_k%d.png' % K)
    plt.colorbar()
    plt.close()
    
    imshow(jnmf.rH1, cmap=cmap, interpolation="nearest", aspect='auto')
    savefig(savedir + '/jnmf_single_rH1_CCLE_k%d.png' % K)
    plt.colorbar()
    plt.close()
    
    imshow(jnmf.rH2, cmap=cmap, interpolation="nearest", aspect='auto')
    savefig(savedir + '/jnmf_single_rH2_CCLE_k%d.png' % K)
    plt.colorbar()
    plt.close()
    
    imshow(jnmf.rH3, cmap=cmap, interpolation="nearest", aspect='auto')
    savefig(savedir + '/jnmf_single_rH3_CCLE_k%d.png' % K)
    plt.colorbar()
    plt.close()
    
    imshow(rX1ori, cmap=cmap, interpolation="nearest", aspect='auto')
    savefig(savedir + '/jnmf_single_rX1_CCLE_k%d.png' % K)
    plt.colorbar()
    plt.close()
    
    imshow(rX2ori, cmap=cmap, interpolation="nearest", aspect='auto')
    savefig(savedir + '/jnmf_single_rX2_CCLE_k%d.png' % K)
    plt.colorbar()
    plt.close()
    
    imshow(rX3ori, cmap=cmap, interpolation="nearest", aspect='auto')
    savefig(savedir + '/jnmf_single_rX3_CCLE_k%d.png' % K)
    plt.colorbar()
    plt.close()
    
    imshow(jnmf.rWH1, cmap=cmap, interpolation="nearest", aspect='auto')
    savefig(savedir + '/jnmf_single_rWH1_CCLE_k%d.png' % K)
    plt.colorbar()
    plt.close()
    
    imshow(jnmf.rWH2, cmap=cmap, interpolation="nearest", aspect='auto')
    savefig(savedir + '/jnmf_single_rWH2_CCLE_k%d.png' % K)
    plt.colorbar()
    plt.close()
    
    imshow(jnmf.rWH3, cmap=cmap, interpolation="nearest", aspect='auto')
    savefig(savedir + '/jnmf_single_rWH3_CCLE_k%d.png' % K)
    plt.colorbar()
    plt.close()


if __name__ == '__main__':
    main()
