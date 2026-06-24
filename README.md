# Reimplementation of NAVI670

## Paper

Mohanty, A. and Gao, G.

"Tightly Coupled Graph Neural Network and Kalman Filter for Smartphone Positioning"

Published in Navigation.

---


## Objective

This repository contains my independent reimplementation of the NAVI670 paper as part of my Master's thesis research.

The goal is to reproduce the reported smartphone GNSS positioning performance using:

- Graph Neural Networks (GraphSAGE)
- Tightly Coupled Bayesian Kalman Filter
- Google Smartphone Decimeter Challenge Dataset

---


## Project Overview

The implementation includes:

- GNSS preprocessing
- Hatch filtering
- Weighted Least Squares initialization
- Graph construction
- GraphSAGE model
- Tightly coupled Bayesian Kalman Filter
- End-to-end training and evaluation

---

## Dataset

Google Smartphone Decimeter Challenge (GSDC)

https://www.kaggle.com/competitions/google-smartphone-decimeter-challenge

Dataset files are not included due to Kaggle licensing restrictions.

---

## Environment

Python 3.11

PyTorch 2.0.1

PyTorch Geometric 2.3.1

---

## Results

Current best result:

| Metric | Paper | Reimplementation |
|----------|----------|----------|
| North Mean Error | 1.9 m | 8.29 m |
| East Mean Error | 1.1 m | 8.63 m |

---

## Current Reproduction Status

Successfully reproduced:

- Data loading pipeline
- Feature extraction
- Graph generation
- GNN architecture
- BKF integration
- Training procedure

Remaining discrepancy:

The reported paper performance has not yet been fully reproduced.

Possible causes:

1. Dataset differences
2. Missing preprocessing details
3. Different initialization strategy
4. Unavailable trained weights
5. Additional filtering not described in paper

---

## Repository Structure

```text
src/
    preprocessing/
    graph/
    models/

results/
    figures/
    tables/

docs/
    reproduction_notes.md
```

---

## Contact

This repository is part of my master's thesis research.

Feedback and suggestions are welcome.
