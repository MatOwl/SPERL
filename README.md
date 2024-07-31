# SPERL

## Quick start
This is the scripts supplementing a paper submission to ICAIF'24. To replicate the empirical result on the paper, readers may run the file `RunMain.py` with Python. 

Each file serves the following purpose:
- `RunMain.py`: Run this file to run the experiment (more technical detials are written as comments in this file)
- `main.py`: the main scritps of this project.
- `settings.py`: global parameter defined for the whole project.
- `Agents.py`: classes of agent.
- `environment.py`: classes of MDP environment.
- `algo.py` and `algoPM.py`: classes of RL algorithms 
- `OptExTree.py`: class defined for the tree-structure illustrated in figure 5 and 6 in the main paper 
- `cubes.py`: scritps for plotting figure 3 and 4 in the main paper;
- `plotting.py`: functions related to plotting (except for plotting cubes);
- `tools.py`: functions repeatly used in different scripts;
- `.npy` files: trained models for specific environment setups (more are stored in the folders named as `seedxxx`).

Following, we provide further details on the MV agents and environments. Full implementation can be found in the scripts in this page and detialed descriptions can be found in `Supplementary Material.pdf`. (For EPG variants and SPERL, initializations, learning rates, and trajectory generations are tuned according to our specific environments; please refer to `run-main.py`.)
