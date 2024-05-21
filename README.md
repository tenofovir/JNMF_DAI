# JNMF_DAI
We propose a novel pipeline integrating joint non-negative matrix factorization (JNMF), identifying key features within sparse high-dimensional heterogeneous data, and a biological pathway analysis, interpreting the functionality of features by detecting activated signaling pathways.
![figure1](https://github.com/tenofovir/JNMF_DAI/assets/90752201/8ef210c8-4d50-4116-b1fe-dd0171305bdd)

# Data
We upload the input data for your convenience, please download and replace the path in the file labeled (run this) with your local path.
Please note that due to the file size limitation, we selected the first 3000 genes of the gene expression file, if you want the complete data, please follow the steps in the manuscript to download the original data in DepMap.

### Code
The simulation data section is available in the "simulate" folder, simply run
```
calc_TriNMF_SimulatedData_binary_missing.py
```
To test the main section, please download the data and all the code in the main section, then run 
```
calc_select_k.py
``` 
or　
```
calc_HexaNMF_CCLE_mask_multi.py
```
