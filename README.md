# Bayesian and Gaussian Process regression (and some examples)
Comparison of several Regression techniques and Gaussian Process.

## Contents

### Examples of classic Bayesian models]:
 - Linear regression
    - Robust linear regression
 - Logistic regression
 - Multivariate Linear and Logistic regression
 - Poisson regression (ZIP)
 - Polynomial regression (univariate and multivariate)
 - Linear splines

### Examples of Gaussian Processes:
 - Gaussian Process Regression
 - Regression with spatial autocorrelation
 - Gaussian Process Classification
    - GP Classification with a More Complex Target
 -  Poisson Process (and Cox Process)

### Application on real data: Yield crop prediction
 - Bayesian models
 - Gaussian processes (potato)
 - Gaussian processes (wheat)

## Repo structure

    .
    ├── data                    # Datasets used in notebooks
    ├── world                   # Geodata for plotting world maps
    ├── 1.* ... 5.*.ipynb       # Notebooks
    ├── guassian_processes.py   # GP utility class
    ├── utils.py                # Utility functions
    ├── LICENSE
    └── README.md
    
## Installiation

Recommended to create a venv; recommended to install pymc separatly with conda if on windows 
[(Instructions)](https://github.com/pymc-devs/pymc/wiki/Installation-Guide-(Windows));

then:

    pip install  -r requirements.txt
    
## Usage

Just use them as a regular ipy notebooks
