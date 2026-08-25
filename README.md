This repository contains all the code to perform an MSSA analysis, as well as a test case notebook to get started.

It has been modeled on the mssa.mat from this Matlab toolkit: https://fr.mathworks.com/matlabcentral/fileexchange/160471-multichannel-singular-spectrum-analysis-significance-test  
The plotting part is highly inspired from Manta et al. (2024) https://www.nature.com/articles/s41598-024-62089-w

These following Python scripts are used to perform an MSSA analysis on a timeserie:

* preprocessing_embedding.py contains the functions to perform the preprocessing of the data (mainly the shaping and the PCA to reduce the dimensionality and the first step of the MSSA that is called embedding, where we create the big trajectory matrix containing all the lagged version of our initial dataset)
* mssa_PC_RC.py contains the functions to actually perform the MSSA, that is calculating the Principal Components (PCs) and the Reconstructed Components (RCs)
* plotting_mssa.py contains the functions allowing the different plots (such as the eigenvalues spectrum, the PCs or RCs timeseries, the RCs maps)
* monte_carlo.py contains the functions to perform the Monte Carlo test (following ???????? Add ref), in order to identify if a mode is significant or not
* tools_MSSA.py contains the functions that are useful for different operations, or that are applying MSSA for different purposes (saving, detrending etc)
* 
The Jupyter notebook mssa_test_case.ipynb present a typical example of how to conduct from scratch an MSSA analysis on a netcdf file, and especially to remove the trend identify by the MSSA from the initial timeserie.
