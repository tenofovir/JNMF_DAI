'''
Created on 2022/05/11

@author: YUTONG DAI
'''

import matplotlib.pyplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.cm as cm
'''
    setting four co-modules for compound data
            three co-modules for expression data
            three co-modules for mutation data
'''
w = np.ones((1, 10))
h1 = np.ones((1, 30))
h2 = np.ones((1, 40))
h3 = np.ones((1, 50))

W = np.zeros((45, 4))  # 5
H1 = np.zeros((4, 130))  # 10
H2 = np.zeros((4, 170))  # 10
H3 = np.zeros((4, 215))  # 15

W[0:10, 0] = w
W[10:20, 1] = w
W[20:30, 2] = w
W[30:40, 3] = w

H1[0, 0:30] = h1
H1[1, 30:60] = h1
H1[2, 60:90] = h1
H1[3, 90:120] = h1

H2[0, 0:40] = h2
H2[1, 40:80] = h2
H2[2, 80:120] = h2
    # H2[3,120:160]=h2

H3[1, 0:50] = h3
H3[2, 50:100] = h3
    # H3[2,100:150]=h3
H3[3, 100:150] = h3

    # original matrix
X1 = np.dot(W, H1)
X2 = np.dot(W, H2)
X3 = np.dot(W, H3)
cmap = cm.Purples
imshow(X1, cmap= cmap, interpolation="nearest", aspect='auto')
matplotlib.pyplot.show()
# add noise
alpha = 0.4
XX1 = X1 + alpha * np.random.randn(45, 130)
XX2 = X2 + alpha * np.random.randn(45, 170)
# XX3 = X3 + beta + alpha * np.random.rand(45, 215)
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
XX1[XX1 > 1] = 2 - XX1[XX1 > 1]
XX2[XX2 > 1] = 2 - XX2[XX2 > 1]
XX3[XX3 > 1] = 2 - XX3[XX3 > 1]

XX1[XX1 < 0] = -XX1[XX1 < 0]
XX2[XX2 < 0] = -XX2[XX2 < 0]
XX3[XX3 < 0] = -XX3[XX3 < 0]
# remove the original negative value which bigger than 1
XX1[XX1 > 1] = 2 - XX1[XX1 > 1]
XX2[XX2 > 1] = 2 - XX2[XX2 > 1]
XX3[XX3 > 1] = 2 - XX3[XX3 > 1]
# ensure that every value in matrix is positive
XX1[XX1 < 0] = -XX1[XX1 < 0]
XX2[XX2 < 0] = -XX2[XX2 < 0]
XX3[XX3 < 0] = -XX3[XX3 < 0]
# check the biggest value is 1
XX1[XX1 > 1] = 1
XX2[XX2 > 1] = 1
XX3[XX3 > 1] = 1

# adding missing values / arrays(nan) 10%rate
binaryM1 = np.random.rand(45, 130) > 0.90
XX1[binaryM1] = np.NaN
binaryA2 = np.random.rand(45) > 0.90
XX2[binaryA2] = np.NaN
binaryA3 = np.random.rand(45) > 0.90
XX3[binaryA3] = np.NaN

simX1 = pd.DataFrame(XX1)
simX2 = pd.DataFrame(XX2)
simX3 = pd.DataFrame(XX3)

#imshow(simX1, cmap='Purples', interpolation="nearest", aspect='auto')

#imshow(simX3, cmap= cmap, interpolation="nearest", aspect='auto')
