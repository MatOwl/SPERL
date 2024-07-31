# SPERL

## Quick start
This page contains the scripts supplementing a paper submission to ICAIF'24. To replicate the empirical result on the paper, readers may run the file `RunMain.py` with Python. 

Each file serves the following purpose:
- `RunMain.py`: Run this file to run the experiment (more technical details are written as comments in this file);
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

Following, we provide further details on the MV agents and environments. Full implementation can be found in the scripts on this page and detailed descriptions can be found in `Supplementary Material.pdf`. (For EPG variants and SPERL, initializations, learning rates, and trajectory generations are tuned according to our specific environments; please refer to `run-main.py`.)
