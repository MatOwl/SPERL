import pandas as pd
import numpy as np

def init():
    # Control Panel
    # Parameters

    global FORCE_UPDATE
    FORCE_UPDATE = False
    
    global bSPERL_TABULAR
    bSPERL_TABULAR = False
    global bSPERL_LINEAR
    bSPERL_LINEAR = False
    global bSPERL_SPECIAL_QUA
    bSPERL_SPECIAL_QUA = False
    
    global RANDOM_SEED
    RANDOM_SEED = 0
    
    global FEATURE_ENG
    FEATURE_ENG = [True, True, True, False]
    # [bTime, bConstant, bQuadratic, bTabular]
    
    global EPSILON
    EPSILON = 0.00001
    
    
    global ENV_TYPE
    ENV_TYPE = "PM"
    global DIR_PREFIX
    DIR_PREFIX = ""
    
    global RISKAVERSION
    global DISCOUNTING
    DISCOUNTING = 1
    RISKAVERSION = 1.2
    
    global ACTIONSPACESIZE
    ACTIONSPACESIZE = 11
    global INITIAL_STATES
    INITIAL_STATES = []
    
    
    global ENV_NAME
    ENV_NAME = ""
    global ALGO_NAME
    ALGO_NAME = ""
    
    global B_TRUEGRADIENT
    
    # Generate All Reachable Time-State
    
def ComputeReferenceTree(treeType,
                        env,
                        initialStates):
    global ENV
    ENV = env
    
    global TIMESTATESPACE
    global TIMESTATESPACE_NUM   # a list of sorted arrays of each time, state
    global TIMEDIVIDER
    global TIMESTATESPACESIZE
    TIMEDIVIDER, TIMESTATESPACE = GetAllTimeState(treeType,
                                                  ENV,
                                                  initialStates,
                                                  bStochasticPolicy = True,
                                                  policyChosen = lambda time, state: [1/ACTIONSPACESIZE] * ACTIONSPACESIZE,
                                                 )
    TIMESTATESPACESIZE = len(TIMESTATESPACE)
        
    global LEN_FEATURE
    LEN_FEATURE = len(TIMESTATESPACE) * ACTIONSPACESIZE
    
    global B_PRINTING
    B_PRINTING = False

    

    
def GetAllTimeState(treeType,
                    env,
                    initialStates,
                    bStochasticPolicy = False,
                    policyChosen = lambda time, state: 1,
                    ):

    
    overall = treeType(policyChosen, 
                       env = ENV,
                       initialStates = initialStates,
                       bStochasticPolicy = bStochasticPolicy)
    
    timeStateSpace = list(overall.stateTimeSpaceDict.keys())
    someState = INITIAL_STATES[0]
    stateLength = len(someState)
    dfEntryList = [[0.] * len(timeStateSpace) for i in range(stateLength+1)]
    
    for j, node in enumerate(overall.stateTimeSpaceDict.values()):
        dfEntryList[0][j] = node.height
        for i in range(stateLength):
            dfEntryList[i+1][j] = node.state[i]

    nodeStateDict = {i:dfEntryList[i] for i in range(stateLength + 1)}
    global NODE_DF
    NODE_DF = pd.DataFrame(nodeStateDict)
    
    timeDivider = [0] * (HORIZON + 1)
    time = -1
    for i,timeState in enumerate(timeStateSpace):
        if timeState[0] > time:
            time = timeState[0]
            timeDivider[time] = i

    #print(len(timeStateSpace))
    return timeDivider, timeStateSpace


def roundUp(n):
    accuracy = 6
    multiplier = 10 ** accuracy
    #print("From %.5f, to %.5f" % (n, round(n * multiplier * 10 // 10) / multiplier))
    return round(n * multiplier) / multiplier
    if False:#n > 0:
        return np.floor(n * multiplier * 10 // 10) / multiplier
    else:
        return np.ceil(n * multiplier * 10 // 10) / multiplier

def StateToString(state):
    try:
        output = ("%.4f, " % (np.sign(state[0]) * roundUp(np.abs(state[0]))))
    except TypeError:
        print(type(state))
        raise(TypeError)
        
    for i in state[1:-1]:
        output += ("%.4f, " % (np.sign(i) * roundUp(np.abs(i))))#"%f, " % round(i, accuracy))
        
    return output+"%.4f"% (np.sign(state[-1]) * roundUp(np.abs(state[-1])))


def GetTimeStateIndex(time, state):
    ## order the state in some specific manner.
    if ENV_TYPE == "OptEx":
        criteria = NODE_DF.iloc[:,0] == time
        temp = NODE_DF[criteria].copy()
        criteria = temp.iloc[:,3] == state[2]
        temp = temp[criteria]
        criteria = temp.iloc[:,1] == state[0]
        
        if len(temp.index[criteria]) > 0:
            return temp.index[criteria][0]
        else:
            temp['diff'] = (temp.iloc[:,1] - state[0]).abs()
            criteria = temp['diff'] == temp['diff'].min()
            return temp.index[criteria][0]
            
        
    else:
        if type(state) == type('1'):
            item = (time, state)
        else:
            item = (time, StateToString(state))
        return TIMESTATESPACE.index(item)

def ComputeFeatureFromTSA(time, state, action):
    if bSPERL_TABULAR:
        output = np.zeros(LEN_FEATURE)
        timtStateIndex = GetTimeStateIndex(time, state)
        somePosition = timtStateIndex * ACTIONSPACESIZE + action
        output[somePosition] = 1
    elif bSPERL_SPECIAL_QUA:
        preFeature = np.append(state, action)
        preFeature = np.append(preFeature, np.array([1]))
        output = np.zeros((HORIZON, len(preFeature)))
        output[time] = preFeature
        output = output.flatten()
        
    else:
        output = []
        preFeature = np.append(state, action)
        if ENV_TYPE == "PM":
            preFeature = np.append(preFeature, np.array([time, 1]))
        else:
            preFeature = np.append(preFeature, np.array([1]))
            
        if bSPERL_LINEAR:
            output = preFeature
        else:
            for i in range(len(preFeature)):
                for j in preFeature[i:]:
                    output.append(preFeature[i]*j)
            output = np.array(output)
            #output = np.append(output, np.array([time]))

    return output