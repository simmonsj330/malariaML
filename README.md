# malariaML
### Authors: Nasib Mansour, Lydia San George, James Simmons
The purose of this repository is to develop and effective machine learning (ML) algorithm based off Cai John's Extreme Phenotype Sampling Machine Learning (ESL-ML) respository in order to predict clearance* for malaria when given dosages of pharmaceuticals.  Additionally, this repository aims to assist clinicians, biologist, and other clinical or scientific field experts by providing a useful ML tool and to increase exposure to such methods.

The randomforest.py file analyzes the malaria data in order to predict clearence.

The  `ESL-ML_archieve` contains the R files, which we base our analysis off of, comes from the ESL-ML repository.

The ESL-ML can be found here:
https://github.com/caiwjohn/EPS-ML

## Malaria Data
The data used comes from ESP-ML and each dataset originates from Zhu et al. and Mok et al. respectfully. The datasets are included in the `malaria_data` directory.

In order to effectily run a random forest algorithm, the original malarai date: mok_meta.txt and zhu_meta.txt is converted into .csv files.

## Dependencies
Python3 

Scikit-Learn

## Build Procedure
*python3 rf_mok.py*

*python3 rf_zhu.py*

### Appendix
Clearence - measurement of per volume of plasma in which a substance is completely removed per unit time.

