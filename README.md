# Mean-variance Reinforcement Learning: Time Inconsistency and Optimality Classes
This repository provides the full codes and supplementary materials for the paper: "Navigating the Difficulty of Achieving Global Optimality in Variance-induced Time-inconsistency"

## Quick start
To replicate the empirical results from the paper, readers may run the file `RunMain.py` with Python. 

Each file serves the following purpose:
- `RunMain.py`: Run this script to run the experiment (more technical details are written as comments in this script);
- `main.py`: the main scripts of this project;
- `settings.py`: global parameter defined for the whole project;
- `Agents.py`: classes of agents;
- `environment.py`: classes of MDP environment;
- `algo.py` and `algoPM.py`: classes of RL algorithms;
- `OptExTree.py`: class defined for the tree structure illustrated in Figure 5 and 6 in the main paper;
- `cubes.py`: scripts for plotting Figure 3 and 4 in the main paper;
- `plotting.py`: functions related to plotting (except for plotting cubes);
- `tools.py`: functions repeatedly used in different scripts (such as generating trajectories);
- `.npy` files: trained models for specific environment setups (more are stored in the `seedxxx` folders).

More details can be found in `Supplementary Material.pdf`. For all EPG variants and SPERL, initializations, learning rates, and trajectory generations are tuned according to the specific environments; please refer to `run-main.py`.

If you find any information from this repo helpful, please cite our work:
```
@inproceedings{10.1145/3677052.3698657,
author = {Tang, Jingxiang and Lesmana, Nixie S and Pun, Chi Seng},
title = {Navigating the Difficulty of Achieving Global Optimality under Variance-Induced Time Inconsistency},
year = {2024},
isbn = {9798400710810},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3677052.3698657},
doi = {10.1145/3677052.3698657},
abstract = {Measuring the long-term uncertainty of following a policy is important and insightful in many risk-sensitive applications such as in finance. In the context of reinforcement learning (RL), risk-based decision-making is commonly incorporated through an RL agent’s goal to optimize the trade-off between the mean and variance of cumulative rewards. However, variance-based objectives are known to induce time inconsistency (TIC). This paper aims to see how TIC permeates into the design and behavior of mean-variance (MV) agents. We approach this by zooming into two optimality classes under TIC: global optimality and equilibrium, for each of which we identify a TIC-aware MV RL method, respectively, episodic policy gradient (EPG) and subgame perfect equilibrium RL (SPERL). We position both methods as approximate tools for achieving global optimality and evaluate their performance in two discerning financial environments: portfolio management and optimal execution. Noteworthily, our results show that despite EPG’s globally optimal objective, its policy does not necessarily attain global optimality and is dominated by equilibrium/SPERL’s policy in numerous environment setups.},
booktitle = {Proceedings of the 5th ACM International Conference on AI in Finance},
pages = {686–694},
numpages = {9},
keywords = {Markov decision processes, agent models, mean-variance optimization, reinforcement learning, time inconsistency},
location = {Brooklyn, NY, USA},
series = {ICAIF '24}
}
```
