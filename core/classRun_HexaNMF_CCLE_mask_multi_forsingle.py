'''
Created on 2022/9/120

@author: dai yutong
'''

from aml.classHexaNMF_mask_rename import *
from aml.classHexaConsensusMatrix_aml import *
from aml.classSaveData_of_HexaNMF_aml import *


class Run_HexaNMF(object):
    '''
    classdocs
    '''

    def __init__(self, X1, X2, X3, X4, X5, X6, maskX1, maskX2, maskX3, maskX4, maskX5, maskX6, K, maxiter, nloop):
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
        self.K = K
        self.maxiter = maxiter
        self.nloop = nloop

    def runNMF_singlerun(self, K, iloop):
        self.K = K
        """
        Run Joint NMF
        """
        report = pd.DataFrame()
        jnmf = HexaNMF_mask(self.X1, self.X2, self.X3, self.X4, self.X5, self.X6, self.maskX1, self.maskX2, self.maskX3,
                            self.maskX4, self.maskX5, self.maskX6, self.K, self.maxiter)
        jnmf.check_nonnegativity()
        jnmf.check_samplesize()
        jnmf.initialize_W_H()
        # jnmf.update_euclidean_multiplicative()
        report = jnmf.wrapper_calc_euclidean_multiplicative_update()
        jnmf.print_distance_report_of_HW_to_X(i)
        # jnmf.set_PanH()
        return report

    def runNMF_multirun(self, K, nloop):
        self.K = K
        self.nloop = nloop

        """
        Run Joint NMF
        Calculate consensus matrix
        """
        cmatrix = HexaConsensusMatrix(self.X1, self.X2, self.X3, self.X4, self.X5, self.X6)
        sdata = SaveHexaDataWH(self.X1, self.X2, self.X3, self.X4, self.X5, self.X6, self.K)
        bestdata = SaveBestHexaDataWH(self.X1, self.X2, self.X3, self.X4, self.X5, self.X6, self.K)
        report = pd.DataFrame()

        # p = Pool(self.nloop)
        # jnmfs = p.map(MulHelper(self, 'runNMF_singlerun'), range(self.nloop))

        for i in range(self.nloop):
            # jnmf = jnmfs[i]
            jnmf = self.runNMF_singlerun(i)
            report.append(jnmf.get_distance_report_of_HW_to_X(i))
            connW = cmatrix.calcConnectivityW(jnmf.W)
            connH1 = cmatrix.calcConnectivityH(jnmf.H1)
            connH2 = cmatrix.calcConnectivityH(jnmf.H2)
            connH3 = cmatrix.calcConnectivityH(jnmf.H3)
            connH4 = cmatrix.calcConnectivityH(jnmf.H4)
            connH5 = cmatrix.calcConnectivityH(jnmf.H5)
            connH6 = cmatrix.calcConnectivityH(jnmf.H6)
            connPanH = cmatrix.calcConnectivityH(jnmf.PanH)
            cmatrix.addConnectivityMatrixtoConsensusMatrix(connW, connH1, connH2, connH3, connH4, connH5, connH6,
                                                           connPanH)
            sdata.save_W_H(jnmf.W, jnmf.H1, jnmf.H2, jnmf.H3, jnmf.H4, jnmf.H5, jnmf.H6, K, i)
            if jnmf.eucl_dist < bestdata.bestscore:
                bestdata.bestscore = jnmf.eucl_dist
                bestdata.save_W_H(jnmf.W, jnmf.H1, jnmf.H2, jnmf.H3, jnmf.H4, jnmf.H5, jnmf.H6, K, i)

        cmatrix.finalizeConsensusMatrix(2)

        """
        Save output files: W, H1, H2, and H3
        """
        return cmatrix, sdata, bestdata, report


class MulHelper(object):
    def __init__(self, cls, mtd_name):
        self.cls = cls
        self.mtd_name = mtd_name

    def __call__(self, *args, **kwargs):
        return getattr(self.cls, self.mtd_name)(*args, **kwargs)


