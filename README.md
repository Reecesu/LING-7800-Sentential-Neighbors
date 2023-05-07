# LING 7800: Computational Models of Discourse
Final Project Abstract

> *Discourse relation classification is an especially difficult task without explicit context markers (Prasad et al., 2008). Current approaches to implicit relation prediction solely rely on two neighboring sentences being targeted, ignoring the broader context of their surrounding environments (Atwell et al., 2021). In this research, we propose three new methods in which to incorporate context in the task of sentence relation prediction: (1) Direct Neighbors (DNs) (2) Expanded Window Neighbors (EWNs) and (3) Part Smart Random Neighbors (PSRNs). Our findings indicate that neighboring context may be useful in the task of predicting implicit discourse relations; however, context beyond directly neighboring sentences can be harmful in the task of discourse relation classification.*

## Data

Contains a README with information regarding The Penn Discourse Treebank (PDTB). Follow data pre-processing steps in main.ipynb to convert piped files to csv format, and place `pdtb2.csv` in `/data` in order to reproduce our results.

## Notebooks

- `main.ipynb` builds the necessary dataframes, defines training arguments, and trains each model.
- `stats_fun.ipynb` displays all our results and tests for significance.
- `test_data_structure.ipynb` is a sandbox for the EWN(), PSRN(), and directNeighbors() functions. Mock datasets help to understand how each function operates.
- `util.py` contains all the helper functions used in other notebooks.
- `visualizations.ipynb` has both static and dynamic graphs of our results.

## Results

Contains each model run output as a csv to be used in `stats_fun.ipynb`. Corresponding plots from `visualizations.ipynb` can also be found here.

### Folder Structure Tree

```
C:.
│   .gitignore
│   LICENSE
│   README.md
│
├───data
│       pdtb2.csv
│       README.md
│
├───notebooks
│   │   main.ipynb
│   │   stats_fun.ipynb
│   │   test_data_structure.ipynb
│   │   util.py
│   │   visualizations.ipynb
│   │
│   └───__pycache__
│           util.cpython-39.pyc
│
├───results
│   │   Baseline tagged no stopwords.csv
│   │   Baseline tagged.csv
│   │   Baseline.csv
│   │   DN tagged no stopwords.csv
│   │   DN tagged.csv
│   │   EWN tagged no stopwords.csv
│   │   EWN tagged.csv
│   │   EWN.csv
│   │   PSRN tagged no stopwords.csv
│   │   PSRN tagged.csv
│   │   PSRN variance test 1.csv
│   │   PSRN variance test 2.csv
│   │   PSRN variance test 3.csv
│   │   PSRN variance test 4.csv
│   │   PSRN variance test 5.csv
│   │   PSRN.csv
│   │
│   └───plots
│           accuracy.png
│           f1.png
│           macro f1.png
│           precision.png
│           recall.png
│
└───__pycache__
        util.cpython-39.pyc
```
