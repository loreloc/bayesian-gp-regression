# Bayesian and Gaussian Process regression (and some examples)
Comparison of several Regression techniques and Gaussian Process.

## Contents
Examples taken from:

[1] MARTIN, OSVALDO. Bayesian Analysis with Python -: Implement Statistical Modeling and Probabilistic Programming Using pymc3. PACKT Publishing Limited, 2018. 

### [Examples of classic Bayesian models:](1.%20bayesian-learning.ipynb)
 - Linear regression
    - Robust linear regression
 - Logistic regression
 - Multivariate Linear and Logistic regression
 - Poisson regression (ZIP)
 - Polynomial regression (univariate and multivariate)
 - Linear splines

### [Examples of Gaussian Processes](2.%20gaussian-processes.ipynb):
 - Gaussian Process Regression
 - Regression with spatial autocorrelation
 - Gaussian Process Classification
    - GP Classification with a More Complex Target
 -  Poisson Process (and Cox Process)

### Application on real data: Yield crop prediction
 - [Preprocessing](3.%20yields-preprocess.ipynb)
 - [Bayesian models](4.%20yields-regression.ipynb)
 - [Gaussian processes (potato)](5a.%20yields-gp-potatoes.ipynb)
 - [Gaussian processes (wheat)](5b.%20yields-gp-wheat.ipynb)

## Repo structure

    .
    ├── data                    # Datasets used in notebooks
    ├── world                   # Geodata for plotting world maps
    ├── 1.* ... 5.*.ipynb       # Notebooks
    ├── guassian_processes.py   # GP utility class
    ├── utils.py                # Utility functions
    ├── LICENSE
    └── README.md
    
## Installation

Recommended to create a venv; recommended to install pymc separatly with conda if on windows 
[(Instructions)](https://github.com/pymc-devs/pymc/wiki/Installation-Guide-(Windows));

then:

    pip install -r requirements.txt
    
## Usage

Just use them as a regular ipy notebooks
