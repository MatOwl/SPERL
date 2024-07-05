import os,sys
import logging
import datetime
import pandas as pd

import algoPM
import settings
settings.init()
import plotting
import algo
import Agents
from tools import *
from OptExTree import *
from environments import *

#===========================================
actionOption = {'bComputePreComm' : False,
                'bComputeSPE' : True,
                'bComputeSPERL' : True,
                'bComputeEPG' : True,
                'bComputeEPGTU': True
                }

bPlot = False
#==============================================


# =======================
# logging config
# =======================

logFormatter = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)s", datefmt='%m/%d/%Y %I:%M:%S %p')
rootLogger = logging.getLogger()
rootLogger.setLevel("DEBUG")

logPath = "Log/"
if os.path.exists(logPath):
    pass
else:
    os.makedirs(logPath, exist_ok=True)
fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, 'runningLog'))
fileHandler.setFormatter(logFormatter)
fileHandler.setLevel("INFO")
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
consoleHandler.setLevel("INFO")
rootLogger.addHandler(consoleHandler)
# logging config finished



class algoCanidate():
    # a class to facilitate training, loading, and evaluating policies for various algo.
    
    def __init__(self, trainingAlgo, bStochastic, envStr = "PM",  policy = lambda a,b: 0, name = ""):
        self.bStochastic = bStochastic
        self.actionList = []
        if trainingAlgo!= None:
            self.policy, self.name, self.trainingProgress = trainingAlgo(envStr)
        else:
            self.policy = policy
            self.name = name
        self.envStr = envStr
        self.J = []
        self.V = []
        self.U = []

    def ComputeTree(self, env, initialStates = []):

        self.tree = OptExTree(self.policy,
                                 env = env,
                                 initialStates = initialStates,
                                 bStochasticPolicy = self.bStochastic,
                                 bComputeMoment = True,
                                 bComputeChild = True
                                 )
        self.J = []
        self.V = []
        self.U = []
        self.P = []
        self.actionList = []
        self.tree.ComputeVisitProb(self.name in ["EPG","EPG_TU", "Naive"])
        self.ExtractJVU()
        self.CreateTable()
    
    def CreateTable(self):
        t = []
        for i,j in enumerate(settings.TIMEDIVIDER):
            try:
                t += [i] * (settings.TIMEDIVIDER[i+1] - settings.TIMEDIVIDER[i])
            except IndexError:
                break
        d = {
            "index": np.arange(len(self.J)),
            "t": np.array(t),
            "A_"+self.name:np.array(self.actionList),
            "J_"+self.name:np.array(self.J),
            "V_"+self.name:np.array(self.V),
            "U_"+self.name:np.array(self.U),
            "P_"+self.name:np.array(self.P)
        }
        self.table = pd.DataFrame(data=d)

    
    def ExtractJVU(self):
        timeDividers = []

        bProb = False
        counter = 0
        time = -1
        
        for i in range(settings.TIMESTATESPACESIZE):
            key = settings.TIMESTATESPACE[i]
            t = key[0]
            if t == settings.HORIZON:
                break

            if (t != time):
                time = t
                if (t > 0):
                    timeDividers.append(counter)
            counter +=1
            
            timeStateNode = self.tree.stateTimeSpaceDict[key]
            self.P.append(timeStateNode.visitProb)
            if self.bStochastic:
                #try:
                actionP = self.policy(t, timeStateNode.GetState())
                J = timeStateNode.J @ actionP
                M = timeStateNode.M @ actionP
                V = (timeStateNode.M - timeStateNode.J **2)@ actionP
                U = (timeStateNode.J - settings.RISKAVERSION * (timeStateNode.M - timeStateNode.J **2)) @ actionP

                pCorrection = timeStateNode.visitProb if bProb else 1
                """
                except KeyError:
                    J = 0
                    M = 0
                    V = 0
                    U = 0
                    pCorrection = 0
                    actionP = [0,0]
                """

                self.J.append(J * pCorrection)
                self.V.append(V * pCorrection)
                self.U.append(U * pCorrection)
                try:
                    self.actionList.append(actionP @ np.arange(settings.ACTIONSPACESIZE))
                except ValueError:
                    print(actionP)
                    print(np.arange(settings.ACTIONSPACESIZE))
                    raise(ValueError)
            else:
                try:
                    action = self.policy(t, timeStateNode.GetState())
                    try: 
                        action = action[0]
                    except:
                        pass
                    J = timeStateNode.J[action]
                    M = timeStateNode.M[action]

                    V = M - J **2
                    U = J - settings.RISKAVERSION * V

                    pCorrection = timeStateNode.visitProb if bProb else 1
                except KeyError:
                    J = 0
                    M = 0
                    V = 0
                    U = 0
                    pCorrection = 0
                    actionP = [0,0] 
                self.J.append(J * pCorrection)
                self.V.append(V * pCorrection)
                self.U.append(U * pCorrection)
                self.actionList.append(action)


def CreateAgent(AlgoName, envStr = "PM", bContTrain = False, b_TrainForNewSeed = False):
    # a function that handle training (by calling the training function).
    
    trainingAlgoDict = {
        "PreCom":algo.ComputePreCommitmentOptPolicy,
        "SPE":algo.ComputeSPEOptPolicy,
        "SPERL":algo.ComputeSPERLPolicy,  
        "EPG":algo.ComputeEPGPolicy,   
        "EPG_TU":algo.ComputeEPGPolicy   
    }
    settings.B_TRUEGRADIENT = AlgoName == "EPG_TU"
    
    if not settings.bSPERL_TABULAR:
        # if not tabular, algo.xxx cannot handle OptEx, hence change
        # but now focused on tabular, so ignore
        trainingAlgoDict["SPERL"] = algoPM.ComputeSPERLPolicy
    if envStr == "PM":
        # if T = 5,  algoPM cannot handle OptEx, hence only change for EPG
        # if not optEx, then ignore.
        trainingAlgoDict["EPG"] = algoPM.ComputeEPGPolicy   
        trainingAlgoDict["EPG_TU"] = algoPM.ComputeEPGPolicy  
        trainingAlgoDict["SPERL"] = algoPM.ComputeSPERLPolicy
    
    try:
        logging.info("Try loading %s Policy" % AlgoName)
        if b_TrainForNewSeed:
            logging.info("Stopped loading. Training for new seed.")
            raise FileNotFoundError
        if AlgoName == "SPE":
            policy = MakeDetPolicy(np.load("./{}_{}.npy".format(AlgoName, 
                                                                settings.ENV_NAME)))
        elif AlgoName in ("EPG", "EPG_TU"):
            print("./{}{}_{}_{}.npy".format(settings.DIR_PREFIX,
                                            AlgoName, 
                                         settings.ENV_NAME,
                                         settings.ALGO_NAME))
            EPGTheta = np.load("./{}{}_{}_{}.npy".format(settings.DIR_PREFIX,
                                                           AlgoName, 
                                                         settings.ENV_NAME,
                                                         settings.ALGO_NAME)
                                )
            if sum(EPGTheta.shape) < 10:
                print(EPGTheta)
            if settings.FORCE_UPDATE:
                raise FileNotFoundError   # force it to train.
            policy = MakeEPGTestPolicy(EPGTheta)
        else:
            if settings.FORCE_UPDATE:
                raise FileNotFoundError   # force it to train.
            policy = MakeDetPolicy(np.load("./{}{}_{}_{}.npy".format(settings.DIR_PREFIX,
                                                                     AlgoName, 
                                                                   settings.ENV_NAME,
                                                                   settings.ALGO_NAME)
                                          )
                                  )
            
        agent = algoCanidate(trainingAlgoDict[AlgoName] if bContTrain else None, 
                             AlgoName in ("EPG", "EPG_TU"),
                             envStr = envStr,
                             policy = policy,
                             name = AlgoName
                            )
        logging.info("%s Policy Loaded" % AlgoName)
        
    except FileNotFoundError:
        logging.info("%s Policy Not Found. Going to Compute" % AlgoName)
        agent = algoCanidate(trainingAlgoDict[AlgoName],
                             AlgoName in ("EPG", "EPG_TU"),
                             envStr = envStr) 
    
    return agent
    
    
def Main(envStr,
          selectAlgo = actionOption, 
          bPrint = False,
          bPlot = True,
          bNotSave = False,
          bContTraining = False,
          nonZeroCheckAlgoList = ["SPE", "SPERL", "EPG", "EPG_TU"],
          parasForAlgo = {'bTrueGradient' : True,
                          'sFuncForm': "linear"},   # linear, quadratic, tabular, ..., 
          parasForEnv = {'horizon': 5},
          initialState = None,
          b_TrainForNewSeed = False,
          randomSeed = 0,
          PolicyFolder = ''):
    
    settings.RANDOM_SEED = randomSeed   # 111,222,333,444,555,666,777,888,999,1000
    resultLocation = "./seed{}/".format(randomSeed)
    if os.path.exists(resultLocation):
        pass
    else:
        os.makedirs(resultLocation, exist_ok=True)
    
    settings.DIR_PREFIX = PolicyFolder
    envType = {"PM": PM,
               "OptEx": OptimalExecution}[envStr]

    # ====================================
    #   load hyperparameters
    # ====================================
    
    settings.HORIZON =parasForEnv['horizon']
    settings.ACTIONSPACESIZE = 11 if envStr == "OptEx" else 2
    if parasForAlgo['sFuncForm'] == 'tabular':
        settings.FEATURE_ENG[2] = False
        settings.FEATURE_ENG[3] = True
        settings.bSPERL_TABULAR = True
        settings.bSPERL_LINEAR = False
        
    
    # ====================================
    #   Loading Parameters and Setup reference
    # ====================================
    
    logging.info("Compute Reference Tree.")
    
    env = envType(parasForEnv)
    settings.ENV_TYPE = envStr
    if envStr == "PM":
        settings.FEATURE_ENG[0] = True  # time is in state, but still
        settings.ILLIQUIDPERIOD = parasForEnv['illiquidPeriod']
        settings.INTERESTRATE = parasForEnv['interestRate']
        settings.P_DEFAULT = parasForEnv['pDefault']
        settings.P_SWITCH = parasForEnv['pSwitch']
        settings.ENV_NAME = "{}_{}_{}_{}_{}_{}".format(envStr,
                                                             settings.HORIZON, 
                                                             settings.ILLIQUIDPERIOD, 
                                                             settings.INTERESTRATE, 
                                                             settings.P_DEFAULT, 
                                                             settings.P_SWITCH
                                                            )
        settings.ALGO_NAME = parasForAlgo['sFuncForm']
        rHigh = parasForEnv["interestRate"][0]
        rLow = parasForEnv["interestRate"][1]
        ltLow = rLow - (rHigh * parasForEnv['pSwitch'] + rLow * (1 - parasForEnv['pSwitch']))
        ltHigh = rHigh - (rHigh * (1 - parasForEnv['pSwitch']) + rLow * parasForEnv['pSwitch'])
        if initialState == None:
            settings.INITIAL_STATES = [np.array([1] + [0] * parasForEnv['illiquidPeriod'] + [ltLow,0]), ]
                                       #np.array([1] + [0] * parasForEnv['illiquidPeriod'] + [ltHigh,0])]
        else:
            settings.INITIAL_STATES = initialState
    elif envStr == "OptEx":
        settings.FEATURE_ENG[0] = False  # time is in state.
        if "gamma" in parasForEnv.keys():
            settings.ENV_NAME = "{}_{}_{}_{}_{}".format(envStr,
                                                        settings.HORIZON,
                                                        parasForEnv['sigma'],
                                                        parasForEnv['numW'],
                                                        parasForEnv['gamma']
                                                        )
        else:
            settings.ENV_NAME = "{}_{}_{}_{}".format(envStr,
                                                        settings.HORIZON,
                                                        parasForEnv['sigma'],
                                                        parasForEnv['numW']
                                                        )
        settings.ALGO_NAME = parasForAlgo['sFuncForm']
        if initialState == None:
            settings.INITIAL_STATES = [env.reset()]
        else:
            settings.INITIAL_STATES = initialState
    settings.ComputeReferenceTree(OptExTree,
                                 env = env,
                                 initialStates = settings.INITIAL_STATES)
    logging.info("Reference Tree Sucessfully Computed.")
    
    # ====================================
    #   load algos
    # ====================================
    algos = []
    settings.B_PRINTING = bPrint
    if bPrint:
        print(len(settings.TIMESTATESPACE))
        print(settings.TIMEDIVIDER)
    logging.info("Experiment Start with")
    logging.info("Size of Time-State Space = %d, divided as %s" % (len(settings.TIMESTATESPACE), str(settings.TIMEDIVIDER)))   
    
    # load naive method
    algos.append(algoCanidate(None, 
                              True,
                              envStr = envStr,
                              policy = lambda a,b:np.array([1.0/settings.ACTIONSPACESIZE] * settings.ACTIONSPACESIZE),
                              name = "Naive"
                            )
                )
    
    if selectAlgo['bComputePreComm' ]:
        algos.append(CreateAgent("PreCom", envStr = envStr, bContTrain = bContTraining))
    if selectAlgo['bComputeSPE']:
        algos.append(CreateAgent("SPE", envStr = envStr, bContTrain = bContTraining))
    if selectAlgo['bComputeSPERL' ]:
        algos.append(CreateAgent("SPERL", envStr = envStr, bContTrain = bContTraining, 
                                 b_TrainForNewSeed = b_TrainForNewSeed))
    if selectAlgo['bComputeEPG']:
        algos.append(CreateAgent("EPG", envStr = envStr, bContTrain = bContTraining, 
                                 b_TrainForNewSeed = b_TrainForNewSeed))
    if selectAlgo['bComputeEPGTU']:
        algos.append(CreateAgent("EPG_TU", envStr = envStr, bContTrain = bContTraining, 
                                 b_TrainForNewSeed = b_TrainForNewSeed))
    # finished loading
    
    # ====================================
    #   analysing
    # ====================================
    
    tempIndex = 0
    for i in algos:
        logging.info("Computing tree for Algo: %s" %(i.name))
        i.ComputeTree(env, initialStates = settings.INITIAL_STATES)
        if tempIndex != 0:
            columnList = ["index", 
                          "A_"+i.name,
                          "J_"+i.name,
                          "V_"+i.name,
                          "U_"+i.name,
                          "P_"+i.name]
            bigDataFrame = pd.merge(bigDataFrame, i.table[columnList], how="left", on="index")
        else:
            bigDataFrame = algos[0].table
        tempIndex += 1
            
    
    sortByList = ["t", "P_SPE"]
    bigDataFrame = bigDataFrame.sort_values(by = sortByList, ascending = [True, False])
    
    # ====================================
    #   ploting
    # ====================================
    if bPlot:
        x = datetime.datetime.now().strftime("%y%m%d%H%M")
        imgLocation = "./graph/{}_{}/" .format(x, 
                                               settings.ENV_NAME)
        if not bNotSave:
            if os.path.exists(imgLocation):
                pass
            else:
                os.makedirs(imgLocation, exist_ok=True)
        
        
        plotting.PlotVisitation(imgLocation, bigDataFrame, 
                                nonZeroCheckAlgoList = nonZeroCheckAlgoList, 
                                bPlot = bNotSave)
        plotting.plotAction(imgLocation, bigDataFrame, 
                            nonZeroCheckAlgoList = nonZeroCheckAlgoList, 
                            bPlot = bNotSave, 
                            bTminusOneOnly = False)
        #plotting.plotAction(imgLocation, bigDataFrame, nonZeroCheckAlgoList = nonZeroCheckAlgoList, bPlot = bNotSave, bTminusOneOnly = True)
        plotting.plotUtility(imgLocation, bigDataFrame, 
                             nonZeroCheckAlgoList = nonZeroCheckAlgoList, 
                             bPlot = bNotSave, 
                             bTminusOneOnly = False, 
                             bRelativeToBase = True)
        plotting.plotUtility(imgLocation, bigDataFrame, 
                             nonZeroCheckAlgoList = nonZeroCheckAlgoList, 
                             bPlot = bNotSave, 
                             bTminusOneOnly = False, 
                             bRelativeToBase = False)
        plotting.plotUtility(imgLocation, bigDataFrame, 
                             nonZeroCheckAlgoList = nonZeroCheckAlgoList, 
                             bPlot = bNotSave, 
                             bProbAdj = True, 
                             bTminusOneOnly = False, 
                             bRelativeToBase = False)
    
    return algos