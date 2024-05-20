'''
@author: DAI yutong

@created on 2022/07/13

This part is designed to find the optimal rank k
Temporarily removes the ability to save data, which can be restored after determining the optimal k

'''

import os
import time
import pandas as pd
import numpy as np

from classRun_HexaNMF_CCLE_mask_multi import *


if __name__ == '__main__':

    start = time.time()

    """
    Read input data files: X1, X2, X3, X4, X5 and X6
    """
    X1ori = pd.read_csv('/Users/yutongdai/Desktop/paper/hexNMF/hex data_DAI/X1_ic50_normalized_data.csv',
                        header=0, index_col=0, na_values='NaN')
    X1 = X1ori[X1ori.notnull().any(axis=1)]  
    # X1 = X1ori.dropna()    
    # X1 = X1ori[X1ori.notnull().all(axis=1)]    
    maskX1 = 1 - X1.isnull()
    X1 = X1.fillna(0)

    X2ori = pd.read_csv('/Users/yutongdai/Desktop/paper/hexNMF/hex data_DAI/X2_MUT_cbio_normalized_binary_data_Complete.edition.csv',
        header=0, index_col=0, na_values='NaN')
    X2 = X2ori[X1ori.notnull().any(axis=1)]  
    # X2 = X2ori[X1ori.notnull().all(axis=1)]    
    maskX2 = 1 - X2.isnull()
    X2 = X2.fillna(0)

    # X3ori = pd.read_csv('C://Users/taiho/Desktop/desktop/Academic/NMF_Python/11_HexaNMF_CCLE/input_CCLE_linage_binmat_5data_modified.csv',
    X3ori = pd.read_csv('/Users/yutongdai/Desktop/paper/hexNMF/hex data_DAI/X3_input_CCLE_GISTIC_AMP2_504.csv',
                        header=0, index_col=0, na_values='NaN')
    # X3ori = pd.read_csv('/Users/yutongdai/Desktop/paper/hexNMF/hex data_DAI/X2_MUT_cbio_normalized_binary_data_Complete.edition.csv',header=0, index_col=0, na_values='NaN')
    # X3 = X3ori[X1ori.notnull().any(axis=1)]
    
    X3 = X3ori.loc[:, X3ori.notnull().any(axis=0)]
    X3 = X3.loc[X1ori.notnull().any(axis=1), :]  

    # X3 = X3.ix[X1ori.notnull().all(axis=1),:] 
    maskX3 = 1 - X3.isnull()
    X3 = X3.fillna(0)

    # X4ori = pd.read_csv('C://Users/taiho/Desktop/desktop/Academic/NMF_Python/11_HexaNMF_CCLE/input_CCLE_linage_binmat_5data_modified.csv',
    X4ori = pd.read_csv('/Users/yutongdai/Desktop/paper/hexNMF/hex data_DAI/X4_input_CCLE_GISTIC_LOSS2_504.csv',
                        header=0, index_col=0, na_values='NaN')
    # X4ori = pd.read_csv('/Users/yutongdai/Desktop/paper/hexNMF/hex data_DAI/X2_MUT_cbio_normalized_binary_data_Complete.edition.csv',header=0, index_col=0, na_values='NaN')
    # X4 = X4ori[X1ori.notnull().any(axis=1)]
    X4 = X4ori.loc[:, X4ori.notnull().any(axis=0)]
    X4 = X4.loc[X1ori.notnull().any(axis=1), :]  
    # X4 = X4.ix[X1ori.notnull().all(axis=1),:]  
    maskX4 = 1 - X4.isnull()
    X4 = X4.fillna(0)

    # X5ori = pd.read_csv('C://Users/taiho/Desktop/desktop/Academic/NMF_Python/11_HexaNMF_CCLE/input_CCLE_linage_binmat_5data_modified.csv',
    X5ori = pd.read_csv('/Users/yutongdai/Desktop/paper/hexNMF/hex data_DAI/X5_CCLE_EXP_504.csv',
                        header=0, index_col=0, na_values='NaN')
    X5 = X5ori[X1ori.notnull().any(axis=1)]
    # X5 = X5.iloc[X1ori.notnull().any(axis=1),:]   
    # X5 = X5.ix[X1ori.notnull().all(axis=1),:]  
    maskX5 = 1 - X5.isnull()
    X5 = X5.fillna(0)

    X6ori = pd.read_csv('/Users/yutongdai/Desktop/paper/hexNMF/hex data_DAI/X6f_CCLE_tumour_type_504 from Fujita.csv',
                        header=0, index_col=0, na_values='NaN')
    # X6 = X6ori[X1ori.notnull().any(axis=1)]     
    #
    # X6 = X6ori[X1ori.notnull().all(axis=1)]    
    maskX6 = 1 - X6ori.isnull()
    X6 = X6ori.fillna(0)
    print("input over")

    """
    Set parameters
    """
    nloop = 10
    maxiter = 2000
    # optimal K = 40 
    c_store = pd.DataFrame(columns=['RANK', 'CCC'])
    for K in [ 10, 20, 30, 40, 50, 60, 70]:
        print("rank k :", K)

        """
        Set output directory
        """
        savedir = '/Users/yutongdai/Desktop/paper/hexNMF/kselect_test_cell_k%d_n%d_iter%d' % (K, nloop, maxiter)
        if not os.path.exists(savedir):
            os.mkdir(savedir)

        print("set dir over")
        """
        Run Joint NMF
        Calculate consensus matrix
        """
        print("running start")
        print(nloop)
        nmf = Run_HexaNMF(X1, X2, X3, X4, X5, X6, maskX1, maskX2, maskX3, maskX4, maskX5, maskX6, K, maxiter, nloop)
        #report = nmf.runNMF_singlerun(K, nloop)
        cmatrix, sdata, bdata, report = nmf.runNMF_multirun(K, nloop)

        """
        Save output files: W, H1, H2
        """


        #sdata.W.to_csv(savedir + '/jnmf_dataW_CCLE_k%d.csv' % K)
        #sdata.H1.to_csv(savedir + '/jnmf_dataH1_CCLE_k%d.csv' % K)
        #sdata.H2.to_csv(savedir + '/jnmf_dataH2_CCLE_k%d.csv' % K)
        #sdata.H3.to_csv(savedir + '/jnmf_dataH3_CCLE_k%d.csv' % K)
        #sdata.H4.to_csv(savedir + '/jnmf_dataH4_CCLE_k%d.csv' % K)
        #sdata.H5.to_csv(savedir + '/jnmf_dataH5_CCLE_k%d.csv' % K)
        #sdata.H6.to_csv(savedir + '/jnmf_dataH6_CCLE_k%d.csv' % K)
        #bdata.W.to_csv(savedir + '/jnmf_bestW_CCLE_k%d.csv' % K)
        #bdata.H1.to_csv(savedir + '/jnmf_bestH1_CCLE_k%d.csv' % K)
        #bdata.H2.to_csv(savedir + '/jnmf_bestH2_CCLE_k%d.csv' % K)
        #bdata.H3.to_csv(savedir + '/jnmf_bestH3_CCLE_k%d.csv' % K)
        #bdata.H4.to_csv(savedir + '/jnmf_bestH4_CCLE_k%d.csv' % K)
        #bdata.H5.to_csv(savedir + '/jnmf_bestH5_CCLE_k%d.csv' % K)
        #bdata.H6.to_csv(savedir + '/jnmf_bestH6_CCLE_k%d.csv' % K)


        #.to_csv(savedir + '/jnmf_distance_report_CCLE_k%d_n%d_iter%d.csv' % (K, nloop, maxiter))
        """
        Plot consensus matrix
        """

        reorderedcmW, c = cmatrix.reorderConsensusMatrix(cmatrix.cmW)
        new_row = {"RANK": K, "CCC": c}
        c_store = c_store.append(new_row, ignore_index=True)
        imshow(reorderedcmW, cmap='Blues', interpolation="nearest")
        plt.colorbar()
        savefig(savedir + '/jnmf_CCLE_ConsensusMatrixW_k%d.clustered.png' % K)
        plt.close()

        """
        imshow(cmatrix.cmPanH, cmap='Blues', interpolation="nearest")
        plt.colorbar()
        savefig(savedir + '/jnmf_CCLE_ConsensusMatrixPanH_k%d.png' % K)
        plt.close()
        """

        """
        Save output files: cmW, cmPanH
        """

        #cmatrix.cmW.to_csv(savedir + '/jnmf_CCLE_ConsensusMatrixW_k%d.csv' % K)
        #cmatrix.cmPanH.to_csv(savedir + '/jnmf_CCLE_ConsensusMatrixPanH_k%d.csv' % K)
        reorderedcmW.to_csv(savedir + '/jnmf_CCLE_ConsensusMatrixW_k%d.clustered.csv' % K)
        c_store.to_csv(savedir + '/CCC_k%d.csv' % K)
        elapsed_time = time.time() - start
        print("time cost is :")
        print(elapsed_time)