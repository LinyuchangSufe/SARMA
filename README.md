# SARMA
This repository contains the code and results for the real data portion of our SARMA paper. 



We have placed all the relevant real data codes we need in the `main` folder.

- The `BCD_LS.py` (`BCD_MLE`) file implements the block coordinate descent algorithm for minimizing the least squares loss function (negative Gaussian quasi likelihood function) of the SARMA model;

- The `help function for LS.py` (`help function for MLE.py`) provides the basic functions for `BCD_LS.py` (`BCD_MLE`);

- The `tensorOp.py` file contains a set of functions related to tensorization and matricization of matrices;
- The `BIC.py` file chooses $(p,r,s)$ order of SARMA model by Bayesian information criterion;
- The `forecast.py` contains some forecast function of SARMA model; 
- The `MACM.py` is a function about matrix autocorrelation function matrix.

- The `IOLS_VARMA.py` is a iterative LS method for solving VARMA model.



In the `Applicaiton` folder,  

- `FREDMD.csv` is the dataset of FRED-MD, which is download from [FRED-MD: A Monthly Database for Macroeconomic Research- Working Papers - St. Louis Fed (stlouisfed.org)](https://research.stlouisfed.org/wp/more/2015-012);

- The `Empirical example.ipynb` file shows how the content is implemented with the code given above and how it compares to other models.

  

The `requirements.txt` file lists the python packages required by the user.
