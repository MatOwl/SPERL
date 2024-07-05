from main import *
#======================================================================
'''
--------------------------------------------------------------------------------
Hyper-Parameters for Algorithm 1 MVarEPG: Mean-Variance Globally Optimal Control
--------------------------------------------------------------------------------
MDP simulator: 
`SampleTrajectory` in tools.py
    - Envonment parameters:
        + OE-1, OE-2: ref to class `OptimalExecution` in environment.py
        + PM-1, PM-2: ref to class `PM` in environment.py
        + specific values are given in the class definition or below.
    - Exploration
        + EPG is stochastic in nature.

lambda: 
`RISKAVERSION` in settings.py

Horizon, T:
`parasForEnv['horizon']` defined below,

Learning rate, alpha and beta:
`fastStepSize`, `slowStepSize` defined in class `EPG` in Agent.py
Values are assigned to them with method `SetStepSize` of class `EPG` in algo.py and algoPM.py

--------------------------------------------------------------------------------
Other parameters about EPG:
--------------------------------------------------------------------------------
episilon in footnote 4 (probability distribution over action space):
`EPSILON` in settings.py

--------------------------------------------------------------------------------
Hyper-Parameters for Algorithm 2 MVarPE: Mean-Variance Policy Evaluation
Algorithm 3 MVarSPERL: Mean-Variance Equilibrium Control
--------------------------------------------------------------------------------
MDP simulator: 
`SampleTrajectoryNoAgent` in tools.py
    - Envonment parameters:
        + OE-1, OE-2: ref to class `OptimalExecution` in environment.py
        + PM-1, PM-2: ref to class `PM` in environment.py
        + specific values are given in the class definition or below.
    - Exploration:
        + epsilon-greedy exploration
        + epsilon: `exploreRate` in `ComputeSPERLPolicy` in algo.py
        + epsilon: `expRate` in `ComputeSPERLPolicy` in algoPM.py

lambda: 
`RISKAVERSION` in settings.py

learning rate, alpha_J, alpha_M:
`stepSize_J_M` in function `ComputeSPERLPolicy` in algo.py, and
`stepSize_J_M` in function `ComputeSPERLPolicy` in algoPM.py

Horizon, T:
`parasForEnv['horizon']` defined below,
'''
#======================================================================
# if the policy does not converge after the specified number of eps,
# set the following flag to be true.
settings.FORCE_UPDATE = False

# specify which methods are used
actionOption = {'bComputePreComm' : False,
                'bComputeSPE' : True,      # True SPE
                'bComputeSPERL' : True,    # SPERL
                'bComputeEPG' : True,   # Approximate EPG
                'bComputeEPGTU': True   # True EPG
                }

# specify the environment
# PM: "PM", OE: "OptEx";
env = "PM"

for rdSeed in [111]:   
    # random seed, choose from [111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]
    if env == 'OptEx':
        result = Main(env,
                      selectAlgo = actionOption, 
                      parasForAlgo = {'sFuncForm': "tabular"},  # only "tabular" is accepted;
                      parasForEnv = {'horizon': 5,              # horizon, T, fixed at 5;
                                    'sigma' : 0.015,            # OE-1:　0.029, OE-2: 0.015;
                                    'numW':4,                   # Fixed at 4;
                                    },
                      randomSeed = rdSeed,
                      PolicyFolder = 'seed%d\\' % reSeed
                      )
        graphPath = "graph/OptEx/"
    elif env == 'PM':
        result = Main(env,
                      selectAlgo = actionOption, 
                      parasForAlgo = {'sFuncForm': "tabular"},  # only "tabular" is accepted;
                      parasForEnv = {"interestRate": [2, 1.1,  1],  # interest rate, [higher rl^nl, lower rl^nl, r^l];
                                                                     # takes value [2, 0.75,  1](PM-1) or [2, 1.1,  1](PM-2)
                                     "horizon" : 5,                  # horizon, T, fixed at 5;
                                     "illiquidPeriod" : 3,           # period of illiquid asset, N, fixed at 3;
                                     "pDefault" : 0.7,               # default probablity of illiquid asset, p_risk;
                                                                     # PM-1: 0.3, PM-2: 0.7;
                                     "pSwitch" : 0.4},               # probablity of switching interest rate, p_switch;
                                                                     # PM-1: 0.2, PM-2: 0.4;
                      randomSeed = rdSeed,
                      PolicyFolder = 'seed%d\\' % rdSeed
                      )
        graphPath = "graph/PM/"
    
    
    
    
# ===============================
# Evalution and Plot
# ===============================

if os.path.exists(graphPath):
    pass
else:
    os.makedirs(graphPath, exist_ok=True)

plotting.DrawTree(result,graphPath, 1, "SPE", (1, "SPE"), bSPEAction = True)
plt.title("TrueSPE")
plt.savefig(graphPath + "SPE_vs_Spe(Action).png",bbox_inches='tight')
plt.show()

plotting.DrawTree(result,graphPath, 4, "EPG_TU", (1, "SPE"), bSPEAction = True)
plt.title("TrueEPG")
plt.savefig(graphPath + "TrueEPG_vs_Spe(Action).png",bbox_inches='tight')
plt.show()

plotting.DrawTree(result,graphPath, 2, "SPERL", (1, "SPE"), bSPEAction = True)
plt.title("SPERL")
plt.savefig(graphPath + "SPERL_vs_Spe(Action).png",bbox_inches='tight')
plt.show()

plotting.DrawTree(result,graphPath, 3, "EPG", (1, "SPE"), bSPEAction = True)
plt.title("ApproxEPG")
plt.savefig(graphPath + "EPG_vs_Spe(Action).png",bbox_inches='tight')
plt.show()


# ===============================
# Obtain and Plot the cubes
# ===============================

import cubes