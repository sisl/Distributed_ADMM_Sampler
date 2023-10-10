# Distributed Markov Chain Monte Carlo Sampling based on the Alternating Direction Method of Multipliers

This repository serves as the code base for the paper "Distributed Markov Chain Monte Carlo Sampling based on the Alternating Direction Method of Multipliers".

The repository contains the following files.

---
`bayesian_lin_reg.py`

We implement D-SGLD and D-SGHMC for a distributed system of agents according to the paper

    "Decentralized Stochastic Gradient Langevin Dynamics and Hamiltonian Monte Carlo",

and D-ULA according to the paper

    "A Decentralized Approach to Bayesian Learning"  

for the Bayesian linear regression set of experiments. 

In addition, we implement our proposed D-ADMMS sampling scheme for the same set of experiments.

---
`bayesian_log_reg.py`

We implement D-SGLD and D-SGHMC for a distributed system of agents according to the paper

    "Decentralized Stochastic Gradient Langevin Dynamics and Hamiltonian Monte Carlo",

and D-ULA according to the paper

    "A Decentralized Approach to Bayesian Learning"
    
for the Bayesian logistic regression set of experiments. 

In addition, we implement our proposed D-ADMMS sampling scheme for the same set of experiments.

---
`Ablation/`

This directory contains the necessary code for the ablation study of D-ADMMS in the context of Bayesian linear
regression.


