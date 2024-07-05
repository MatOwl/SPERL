import numpy as np
import pandas as pd

import settings
import Agents
from tools import *
from OptExTree import *
from environments import *

import logging

logger = logging.getLogger('root')

def ComputeSperlQWithFeature(feature, paraJ, paraM):
    # output: Q is a scaler
    if settings.bSPERL_TABULAR:
        J = paraJ @ feature
        M = paraM @ feature
        Q = J - settings.RISKAVERSION * (M - J **2)
    else:
        # paraJ: T x n
        # J = （T,n) * (n * 1)
        J = paraJ @ feature
        M = paraM @ feature
        Q = J - settings.RISKAVERSION * (M - J **2)
    return Q    # scaler


def PolicyEvlTDZero(eps, 
                    env, 
                    policyChosen, 
                    bStochasticPolicy, 
                    initJ, 
                    initM,
                    stepSize = 0.01,
                    selectedTime = -1,   # (settings.HORIZON - 1) to 0 [reversed(range(settings.HORIZON))]
                    trajList = [], 
                    reuseNumber = 0,
                    refU = np.array([]),
                    treeRef = None
                   ):
    
    
    paraVectorJ =initJ.copy()
    paraVectorM =initM.copy()
    if settings.bSPERL_TABULAR:
        initU = paraVectorJ - settings.RISKAVERSION * (paraVectorM - paraVectorJ**2)   
    elif treeRef != None:
        initU = np.zeros(refU.shape)
        for i in range(settings.TIMESTATESPACESIZE):
            key = settings.TIMESTATESPACE[i]
            if key[0] == settings.HORIZON:
                break
            timeStateNode = treeRef.stateTimeSpaceDict[key]
            for a in range(settings.ACTIONSPACESIZE):
                feature = settings.ComputeFeatureFromTSA(timeStateNode.height, timeStateNode.state, a)
                initU[2*i + a] = ComputeSperlQWithFeature(feature, paraVectorJ, paraVectorM)
    else:
        initU = np.zeros(refU.shape)
    
    gap = eps - len(trajList)
    if gap > 0:
        trajList = trajList
        for k in range(gap):
            trajSamp = SampleTrajectoryNoAgent(env, policyChosen, bStochasticPolicy = bStochasticPolicy)
            trajList.append(trajSamp)
            
    #stateTimeFrequency = np.zeros(settings.LEN_FEATURE)# debug

    ## actual simulation
    for k in range(eps):
        errorArray = []   # for debug
        
        correctionTermJ = np.zeros(initJ.shape)
        correctionTermM = np.zeros(initM.shape)
        counterTemp = 0
        for traj in trajList[max(0, k - reuseNumber): k+1]:  # [trajSamp]: 
            counterTemp += 1
            feature = settings.ComputeFeatureFromTSA(selectedTime, traj[selectedTime][0], traj[selectedTime][1])
            nextTime = selectedTime+1

            #stateTimeFrequency += feature # debug

            TDTermJ = traj[selectedTime][2] # reward at the selected time
            TDTermM = traj[selectedTime][2]**2   # reward ** 2

            if nextTime == settings.HORIZON:
                # simplified as J and M for last state are always 0
                TDTermJ += 0 - feature @ paraVectorJ
                TDTermM += 0 - feature @ paraVectorM
            else:                    
                if bStochasticPolicy:
                    nextActionP = policyChosen(nextTime, traj[nextTime][0])

                    nextFeature_0 = settings.ComputeFeatureFromTSA(nextTime, traj[nextTime][0], 0)                     
                    nextFeature_1 = settings.ComputeFeatureFromTSA(nextTime, traj[nextTime][0], 1)                     

                    TDTermJ += nextActionP[0] * (settings.DISCOUNTING * nextFeature_0 - feature) @ paraVectorJ
                    TDTermJ += nextActionP[1] * (settings.DISCOUNTING * nextFeature_1 - feature) @ paraVectorJ

                    TDTermM += nextActionP[0] * (2 * settings.DISCOUNTING * traj[selectedTime][2] * nextFeature_0 @ paraVectorJ + 
                                                 (settings.DISCOUNTING**2 * nextFeature_0 - feature) @ paraVectorM)
                    TDTermM += nextActionP[1] * (2 * settings.DISCOUNTING * traj[selectedTime][2] * nextFeature_1 @ paraVectorJ + 
                                                 (settings.DISCOUNTING**2 * nextFeature_1 - feature) @ paraVectorM)

                else:
                    nextAction = policyChosen(nextTime, traj[nextTime][0])
                    nextFeature = settings.ComputeFeatureFromTSA(nextTime, traj[nextTime][0], nextAction)
                    
                    TDTermJ += (settings.DISCOUNTING * nextFeature - feature) @ paraVectorJ
                    TDTermM += (2 * settings.DISCOUNTING * traj[selectedTime][2] * nextFeature @ paraVectorJ + 
                                (settings.DISCOUNTING**2 * nextFeature - feature) @ paraVectorM
                               )
                
            correctionTermJ += feature * TDTermJ
            correctionTermM += feature * TDTermM

        ## update parameters
        paraVectorJ += stepSize * correctionTermJ / ((counterTemp*eps) if not settings.bSPERL_TABULAR else 1)#/eps
        paraVectorM += stepSize * correctionTermM / ((counterTemp*eps) if not settings.bSPERL_TABULAR else 1)# 
        if settings.bSPERL_TABULAR:
            estU = paraVectorJ - settings.RISKAVERSION * (paraVectorM - paraVectorJ**2)   
        elif treeRef != None:
            estU = np.zeros(refU.shape)
            for i in range(settings.TIMESTATESPACESIZE):
                key = settings.TIMESTATESPACE[i]
                if key[0] == settings.HORIZON:
                    break
                timeStateNode = treeRef.stateTimeSpaceDict[key]
                for a in range(settings.ACTIONSPACESIZE):
                    feature = settings.ComputeFeatureFromTSA(timeStateNode.height, timeStateNode.state, a)
                    estU[2*i + a] = ComputeSperlQWithFeature(feature, paraVectorJ, paraVectorM)
        else:
            estU = np.zeros(refU.shape)
        
        if estU[0] > 10000:
            paraVectorJ /= paraVectorJ[0]
            paraVectorM /= paraVectorM[0]
                
    logger.info("Diff = %.5f, Change= %.5f" % (np.sum(np.absolute(estU - refU)), np.sum(np.absolute(estU - initU))))
    return paraVectorJ, paraVectorM# , stateTimeFrequency

class SPERLDataPack():
    def __init__(self):
        self.Keys = []
        self.ActionDict = {}
        self.JEstDict = {}
        self.MEstDict = {}
    
    def GetAction(self, time, state):
        pass
        


def ComputeSPERLPolicy(envStr = "PM"):
    # For debug purpose (checking answer)
    SPEPolicy = MakeDetPolicy(np.load("./{}_{}.npy".format("SPE", 
                                                           settings.ENV_NAME))
                             )
    settings.ENV.reset()
    randomSeed = settings.RANDOM_SEED
    np.random.seed(randomSeed)
    
    tempSPETree =  OptExTree(SPEPolicy,
                             env = settings.ENV,
                             initialStates = settings.INITIAL_STATES,
                             bStochasticPolicy = False,
                             bComputeMoment = True
                             )
    
    CorrectJ = np.zeros(settings.LEN_FEATURE)
    CorrectM = np.zeros(settings.LEN_FEATURE)
    
    for i in range(settings.TIMESTATESPACESIZE):
        key = settings.TIMESTATESPACE[i]
        if key[0] == settings.HORIZON:
            break
        timeStateNode = tempSPETree.stateTimeSpaceDict[key]
        for a in range(settings.ACTIONSPACESIZE):
            CorrectJ[2*i + a] = timeStateNode.J[a]
            CorrectM[2*i + a] = timeStateNode.M[a]
    
    CorrectU = CorrectJ - settings.RISKAVERSION * (CorrectM - CorrectJ**2)    

    if settings.FORCE_UPDATE:
        fileName = "./seed{}/{}_{}_{}".format(randomSeed,
                                                 "SPERL", 
                                       settings.ENV_NAME,
                                       settings.ALGO_NAME)
        actionList = np.load(fileName + '.npy')
        paraVectorJ = np.load(fileName + '_J.npy')
        paraVectorM = np.load(fileName + '_M.npy')
        trainingPerformance = list(np.load(fileName + '_Performance.npy'))
    else:
        actionList = np.random.randint(2, size = settings.TIMESTATESPACESIZE) #[0] *settings.TIMESTATESPACESIZE  #

        # defining theta_J, theta_M
        paraShape = settings.ComputeFeatureFromTSA(0, settings.ENV.reset(), 0).shape
        if not settings.bSPERL_TABULAR:
            pass
            #paraShape = (paraShape[0]*settings.HORIZON, paraShape[1])
        paraVectorJ = np.zeros(paraShape)
        paraVectorM = np.zeros(paraShape)
        trainingPerformance = []
    
    trainingEpsSPERL = 750 if settings.bSPERL_TABULAR else 200
    maxRound = 150 if settings.ENV_TYPE == "PM" else 500
    if settings.FORCE_UPDATE:
        maxRound = 50
    
    env = settings.ENV
    

    
    oldPolicy = [1] * (settings.TIMESTATESPACESIZE)
    counter = 0
    actionListList = []
    
    policyChosen = MakeDetPolicy(actionList)
    if settings.ENV_TYPE == "PM":
        stepSize_J_M = 0.1  if settings.bSPERL_TABULAR else 0.0005#0.0000005
        stepSize_J_M *= 3 if settings.bSPERL_LINEAR else 1
    else:
        stepSize_J_M = 0.1  if settings.bSPERL_TABULAR else 0.00001#0.0000005
        stepSize_J_M *= 500 if settings.bSPERL_SPECIAL_QUA else 1
        stepSize_J_M *= 3 if settings.bSPERL_LINEAR else 1
    expRate = 0.95
    roundNumber = 5 #if settings.bSPERL_TABULAR else 10
    
    print(trainingEpsSPERL)
    while (not (np.all(actionList == oldPolicy) and counter > 30 ) and counter < maxRound):
        if counter % roundNumber == roundNumber-1:
            if settings.FORCE_UPDATE:
                expRate = 0.
            else:
                expRate *= 0.9

        oldPolicy = actionList.copy()
        policyExplore = MakeStochasticExplorationPolicy(actionList.copy(), expRate)
        
        trajList = [[]] * trainingEpsSPERL
        for k in range(trainingEpsSPERL):
            Done = False
            while not Done:
                try:
                    trajSamp = SampleTrajectoryNoAgent(env, policyExplore, bStochasticPolicy = False)
                    trajList[k] = trajSamp
                    Done = True
                except ValueError:
                    pass
                    #logger.warning("Not found In List again at %d" % k)
                    
        
        for time in reversed(range(settings.HORIZON)):
            policyChosen = MakeDetPolicy(actionList.copy())
            paraVectorJ, paraVectorM = PolicyEvlTDZero(trainingEpsSPERL, 
                                                          env, 
                                                          policyChosen, 
                                                          False, 
                                                          paraVectorJ,
                                                          paraVectorM,
                                                          stepSize = stepSize_J_M,
                                                          selectedTime = time,
                                                          trajList = trajList,
                                                          refU = CorrectU,
                                                          treeRef = tempSPETree
                                                )

            # improvement:
            for timeStateIndexI in range(settings.TIMESTATESPACESIZE):
                if settings.TIMESTATESPACE[timeStateIndexI][0] < time:
                    continue
                elif settings.TIMESTATESPACE[timeStateIndexI][0] > time:
                    break
                
                # Q（s,t,a）应该被定义为根据feature的一个函数，目前事最 naive 的办法，查表。
                Qarray = [0.] * settings.ACTIONSPACESIZE
                key = settings.TIMESTATESPACE[timeStateIndexI]
                timeStateNode = tempSPETree.stateTimeSpaceDict[key]
                for action in range(settings.ACTIONSPACESIZE):
                    featureTemp = settings.ComputeFeatureFromTSA(timeStateNode.height, timeStateNode.state, action)

                    Qarray[action] = ComputeSperlQWithFeature(featureTemp, paraVectorJ, paraVectorM)

                # 然后 greedy。
                actionList[timeStateIndexI] = np.argmax(Qarray)
                

        tempPolicy = MakeDetPolicy(actionList.copy())
        tempTree = OptExTree(tempPolicy,
                             env = settings.ENV,
                             initialStates = settings.INITIAL_STATES,
                             bStochasticPolicy = False,
                             bComputeMoment = True
                             )
        JTest = [0]
        VTest = [0]
        for rootNode in tempTree.roots:
            actionP = np.zeros(settings.ACTIONSPACESIZE)
            action = int(tempPolicy(0, rootNode.GetState()))
            actionP[action] = 1
            JTest[0] += (rootNode.J @ actionP) / len(tempTree.roots)
            VTest[0] += (rootNode.M - rootNode.J **2)@actionP / len(tempTree.roots)


        logger.info("Testing Result for this round: Mean = %.5f, Var = %.5f, Utility = %.5f ." % 
                    (JTest[-1], 
                     VTest[-1], 
                     JTest[-1] - settings.RISKAVERSION * VTest[-1]))
        
        actionListList.append(actionList.copy())
        trainingPerformance.append(JTest[-1] - settings.RISKAVERSION * VTest[-1])
        
        outputAL = actionListList[-1]
        logger.info("Saving %s Policy" % "SPERL")
        fileName ="./seed{}/{}_{}_{}.npy".format(randomSeed,
                                                 "SPERL", 
                                       settings.ENV_NAME,
                                       settings.ALGO_NAME)
        np.save(fileName, outputAL.copy())
        fileName ="./seed{}/{}_{}_{}_J.npy".format(randomSeed,
                                                 "SPERL", 
                                       settings.ENV_NAME,
                                       settings.ALGO_NAME)
        np.save(fileName, paraVectorJ.copy())
        fileName ="./seed{}/{}_{}_{}_M.npy".format(randomSeed,
                                                 "SPERL", 
                                       settings.ENV_NAME,
                                       settings.ALGO_NAME)
        np.save(fileName, paraVectorM.copy())
        

        counter += 1
    
    logger.info("Finished computing SPERL Policy")
    if counter == maxRound:
        logger.warning("SPERL policy does not converge.")
    
    outputAL = actionListList[-1]
    logger.info("Saving %s Policy" % "SPERL")
    fileName ="./seed{}/{}_{}_{}_J.npy".format(randomSeed,
                                                 "SPERL", 
                                       settings.ENV_NAME,
                                       settings.ALGO_NAME)
    np.save(fileName, paraVectorJ.copy())
    fileName ="./seed{}/{}_{}_{}_M.npy".format(randomSeed,
                                             "SPERL", 
                                   settings.ENV_NAME,
                                   settings.ALGO_NAME)
    np.save(fileName, paraVectorM.copy())
    fileName ="./seed{}/{}_{}_{}.npy".format(randomSeed,
                                             "SPERL", 
                                     settings.ENV_NAME,
                                      settings.ALGO_NAME)
    np.save(fileName, outputAL)
    if trainingPerformance:
        fileName ="./seed{}/{}_{}_{}_Performance.npy".format(randomSeed,
                                                             "SPERL",
                                                      settings.ENV_NAME,
                                                      settings.ALGO_NAME)
        np.save(fileName, trainingPerformance)
    logger.info("%s Policy Saved at %s ." % ("SPERL", fileName))
    
    #return oeAgent, JMTable
    return MakeDetPolicy(outputAL), "SPERL", trainingPerformance

def ComputeEPGPolicy(envStr = "PM"):
    # ============== Main =======================
    # Train and test combined and repeat    
    
    def Main(agent, env, numOfRound = 100, trainEps = 100, testEps = 0, ssQueue = [10000, 1000000]):
        meanList = []
        varList = []
        thetaList = []
        wealthList = []
        wealthVarList = []
        trainingPerformance = []
        ss = ssQueue[0]
        oldU = -100
        
        #for i in range(numOfRound):       
        i  = -1
        while True:
            i += 1
            bStepSizeTooLarge = False
            
            # EPG method
            #MCTrain(trainEps, env, agent, bTrain = True, debug = False, report = False)
            # if i == numOfRound//2:
            
            if settings.B_TRUEGRADIENT:   
                # true update by nabla
                agent.SetStepSize(ss, ss, 0)
                agent.ThetaBackUp()
                print("start true gradient approach")
                for j in range(1):
                    tempPolicy = MakeEPGTestPolicy(agent.theta)
                    tempTree = OptExTree(tempPolicy,
                                         env = settings.ENV,
                                         initialStates = settings.INITIAL_STATES,
                                         bStochasticPolicy = True,
                                         bComputeMoment = True
                                         )
                    
                    trueJ = 0
                    trueU = 0
                    for rootNode in tempTree.roots:
                        actionP = tempPolicy(0, rootNode.GetState())
                        trueJ += (rootNode.J @ actionP)
                        trueU += (rootNode.J - settings.RISKAVERSION * (rootNode.M - rootNode.J**2))@ actionP
                    trueJ /= len(tempTree.roots)
                    trueU /= len(tempTree.roots)
                    
                    nabla_J, Prenabla_V = tempTree.ComputePolicyGradient(agent.theta, agent.zUpdate)
                    nabla_V = Prenabla_V - 2 * trueJ * nabla_J
                    #print(nabla_J[0][:5])
                    #print(nabla_V[0][:5])
                    #nabla_J, nabla_V = ComputeNablaBySimulation(10000, env, agent, trueJ)
                    #print(nabla_J.shape)
                    #print(nabla_J[:5])
                    #print(nabla_V[:5])
                    
                    agent.TrueUpdate(nabla_J, nabla_V)
                if (oldU - trueU) > 0:
                    # stepsize too large, bad things happended
                    agent.ThetaRollBack()
                    
                    logger.info("Rollback happened. stepSize Updated: %.2f" % (ss))
                    ss *= 0.1
                    if ss <= (0.001 if envStr == "PM" else .000001):
                        bStepSizeTooLarge = True
                        break
                    i -= 1
                    continue
                print(trueU)
                trainingPerformance.append(trueU)
                if np.abs(oldU - trueU) < 0.000001:
                    break
                    
                oldU = trueU
                logger.info("%d: %.7f, %.7f" % (i, trueU, np.sum(nabla_J)))
            else:
                if i == 800:   # number of iteration for approximate gradient
                    break
                agent.SetStepSize(0.005, 0.0001, 0)
                if i % 10 == 9:
                    tempPolicy = MakeEPGTestPolicy(agent.theta)
                    tempTree = OptExTree(tempPolicy,
                                         env = settings.ENV,
                                         initialStates = settings.INITIAL_STATES,
                                         bStochasticPolicy = True,
                                         bComputeMoment = True
                                         )
                    trueJ = 0
                    trueU = 0

                    for rootNode in tempTree.roots:
                        actionP = tempPolicy(0, rootNode.GetState())
                        trueU += (rootNode.J - settings.RISKAVERSION * (rootNode.M - rootNode.J**2)
                                 )@ actionP / len(tempTree.roots)
                    logger.info(str(trueU))
                    trainingPerformance.append(trueU)
                    
                    
                MCTrain(trainEps, env, agent, bTrain = True, debug = False, report = False)
            
            
            
            """
            testMean, testVar, wealthMean, wealthVariance = MCTrain(testEps, 
                                                                    env, 
                                                                    agent, 
                                                                    bTrain = False, 
                                                                    debug = False, 
                                                                    report = False)
            
            meanList.append(testMean)
            varList.append(testVar)
            wealthList.append(wealthMean)
            wealthVarList.append(wealthVariance)
            """
            thetaList.append(agent.theta.flatten())
            if i % 5 == 0:
                fileName ="./seed{}/{}_{}_{}.npy".format(settings.RANDOM_SEED,
                                                         "EPG" + ("_TU" if settings.B_TRUEGRADIENT else ''),
                                                  settings.ENV_NAME,
                                                  settings.ALGO_NAME) 
                np.save(fileName, agent.theta)
            

        return meanList, varList, thetaList, wealthList, wealthVarList, trainingPerformance

    def MakeOEEPGPolicy():
        def policy(theta, state, debug = False):
            # return the probability of each option
            feature = GetEPGFeature(state)
            
            if len(feature) == 1:
                a = -theta * feature
            else:
                a = -theta @ feature

            a = np.exp(a)
            a = a/np.sum(a)

            if np.sum(a[:-1]) <= 1:
                a[-1] = 1 - np.sum(a[:-1])
            else:
                a[-1] = 0.
                ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
                a[ind] -= 1 - np.sum(a[:-1])

            #[invest 0,  invest 0.1, invest 0.2, invest 0.3, invest 0.4,
            # invest 0.5,invest 0.6, invest 0.7, invest 0.8, invest 0.9,
            # invest 1]
            if any(a < 0):
                print(a)
            return a

        return policy

    def MakeOEZUpdate(actionSpace, thetaShape):
        def ZUpdate(theta, state, action):
            feature = GetEPGFeature(state)
            
            if len(feature) == 1:
                a = -theta * feature
            else:
                a = -theta @ feature

            a = np.exp(a)
            temp_origin = a
            a = a/np.sum(a)

            if np.sum(a[:-1]) <= 1:
                a[-1] = 1 - np.sum(a[:-1])
            else:
                a[-1] = 0.
                ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
                a[ind] -= 1 - np.sum(a[:-1])
            
            
            actionIndex = actionSpace.index(action)
            temp = np.zeros(theta.shape)
            for i in range(theta.shape[0]):
                if i == actionIndex:
                    temp[actionIndex] = -(1 - a[actionIndex]) * feature
                else:
                    temp[i] = a[actionIndex] * feature
                    
                        
    
            
            """
            if len(feature) == 1:
                sums = np.sum(theta) * feature
            else:
                sums = np.sum(np.dot(theta, feature))
            temp = np.array([-1 * feature / sums] * thetaShape[0]).reshape(thetaShape)
            temp[actionIndex] += np.copy(feature)
            """
            return temp
        return ZUpdate
    """
    def MakeOEEPGPolicy():
        def policy(theta, state, debug = False):
            # return the probability of each option
            feature = GetEPGFeature(state)
            if len(feature) == 1:
                a = theta * feature
            else:
                a = theta @ feature

            a = np.exp(a)    

            if any(np.isinf(a)):
                a = np.isinf(a) * 1
            if np.sum(a) == 0:
                a = np.array([1]*len(a))

            a = a/max(a)
            temp_origin = a
            a = a/np.sum(a)

            if np.sum(a) != 1:
                temp = a
                a = a/np.sum(a)
                if any(np.isnan(a)):
                    print(temp_origin)
                    print(temp)
                    raise ValueError

            #[invest 0, invest 0.1,invest 0.2,invest 0.3,invest 0.4,invest 0.5,invest 0.6, invest 0.7, invest 0.8,invest 0.9, invest 1]
            return a

        return policy

    def MakeOEZUpdate(actionSpace, thetaShape):
        def ZUpdate(theta, state, action):
            feature = GetEPGFeature(state)
            
            if len(feature) == 1:
                sums = np.sum(theta) * feature
            else:
                sums = np.sum(np.dot(theta, feature))
            actionIndex = actionSpace.index(action)
            temp = np.array([-1 * feature / sums] * thetaShape[0]).reshape(thetaShape)
            temp[actionIndex] += np.copy(feature)
            return temp
        return ZUpdate
    """
    def MakeZUpdate(actionSpace, thetaShape, epsilon):
        def ZUpdate(theta, state, action):
            feature = GetEPGFeature(state)
            
            expTerm = np.exp(-np.dot(theta, feature))
            
            if action == 1:
                nominator = epsilon + (1 - 2* epsilon) / (1+expTerm)
                tempReal = (1 - 2* epsilon) * expTerm / (1+expTerm)**2/ nominator
            else:
                nominator = 1 - epsilon - (1 - 2* epsilon) / (1+expTerm)
                tempReal = -(1 - 2* epsilon) * expTerm/ (1+expTerm)**2/ nominator
            temp = tempReal * feature
            return temp
        return ZUpdate
    
    def MakeEPGPolicy(epsilon):
        def policy(theta, state, debug = False):
            # return the probability of each option
            
            feature = GetEPGFeature(state)
            p = epsilon + (1 - 2 * epsilon)/(1+np.exp(-np.dot(theta, feature)))[0]
            # dot product negative, epsilon
            # dot product positive, 1 - epsilon
            return np.array([1-p, p])

        return policy
    
    
    trainEps = 5000
    testEps = 1
    trainRound = 50
    epsilon = settings.EPSILON
    trainingPerformance = []

    # agent
    env = settings.ENV
    
    if settings.ENV_TYPE == "PM":
        tempInt = 2 + 1 + settings.ILLIQUIDPERIOD
        #thetaSize = [1 ,  ((tempInt + 1) + (tempInt + 1)**2)//2]
        thetaSize = [1, len(GetEPGFeature(env.reset()))]
        print(thetaSize)
        # back up: thetaSize = [1 ,2 + 1 + settings.ILLIQUIDPERIOD]
        actionSpace = [0,1]
        policy = MakeEPGPolicy(epsilon)
        zUpdateFn = MakeZUpdate(actionSpace, thetaSize, epsilon)
    elif settings.ENV_TYPE == "OptEx":
        thetaSize = [settings.ACTIONSPACESIZE, len(GetEPGFeature(env.getState()))]
        print(thetaSize)
        
        actionSpace = [i for i in range(settings.ACTIONSPACESIZE)]
        policy = MakeOEEPGPolicy()
        zUpdateFn = MakeOEZUpdate(actionSpace, thetaSize)

    
    # result plotting
    EFvarianceList = []
    EFwealthList = []
    thetaOutputList = []
    UList = []

    g = settings.RISKAVERSION
    seedList = [settings.RANDOM_SEED] #,325,53,75]

    optU = -100
    optTheta = []
    
    for s in seedList:
        logger.info("Training EPG for seed %d" % s)
        np.random.seed(s)
        # define agent accordingly
        if settings.FORCE_UPDATE:
            initTheta = np.load("./{}_{}_{}.npy".format("EPG"  + ("_TU" if settings.B_TRUEGRADIENT else ''), 
                                      settings.ENV_NAME,
                                      settings.ALGO_NAME)
                               )
        else:
            initTheta =  (np.random.rand(*thetaSize) - 0.5)* 1
        EPGAgent = Agents.EPG(thetaSize, 
                                      actionSpace, 
                                      policy, 
                                      zUpdateFn, 
                                      initTheta = initTheta, 
                                      objectiveType = 1, 
                                      gamma = g, 
                                      epsilon = epsilon)
        # 100 for FA (7 varaibles)
        # 0: 4.8643050, 0.1135740 (100 for 7 horizon)
        # horizon = 5, 10000 first, then 1,000,000,000
        
        # Train and Collect data
        meanList, varList, thetaList, wealth, wealthVar, Performance = Main(EPGAgent, 
                                                                               env,
                                                                               trainRound, 
                                                                               trainEps, 
                                                                               testEps,
                                                                               ssQueue = [500. if envStr == "PM" else 50, 1.])
        trainingPerformance += Performance
        
        if settings.B_PRINTING:
            PlotTrainingProgress(meanList, 
                                  varList, 
                                  thetaList, 
                                 np.log(np.array(wealth)), 
                                 np.log(np.array(wealthVar)), 
                                 0.2, 
                                 thetaSize
                                )
            
        logger.info("Finished Training EPG for seed %d" % s)
        
        logger.info("Evaluating EPG for seed %d" % s)
        """
        JTest, VTest, theta_output, wealth, wealthVar = Main(EPGAgent, env, 1, 0, 5000)
        EFwealthList += wealth
        EFvarianceList += wealthVar
        """
        """
        for i in range(10):
            tradeTraj  = [t[0][2] for t in SampleTrajectory(env, agent)]
            Etemp, Vtemp = EVComputation(tradeTraj)
            simEList.append(Etemp)
            simVList.append(Vtemp)
        """
        # this is not a "legal" way of training, because it uses the true value. Just to debug and test.
        optForThisRound = np.array(thetaList[-1]).reshape(thetaSize)
        tempPolicy = MakeEPGTestPolicy(optForThisRound)
        tempTree = OptExTree(tempPolicy,
                             env = settings.ENV,
                             initialStates = settings.INITIAL_STATES,
                             bStochasticPolicy = True,
                             bComputeMoment = True
                             )
        JTest = [0]
        VTest = [0]
        for rootNode in tempTree.roots:
            actionP = tempPolicy(0, rootNode.GetState())
            JTest[0] += (rootNode.J @ actionP) / len(tempTree.roots)
            VTest[0] += (rootNode.M - rootNode.J **2)@actionP / len(tempTree.roots)
        
        logger.info("Testing Result for this round: Mean = %.5f, Var = %.5f, Utility = %.5f ." % 
                    (JTest[-1], 
                      VTest[-1], 
                      JTest[-1] - settings.RISKAVERSION * VTest[-1]))
        
        #logger.info("Current Theta = %s" % (str(optForThisRound)))
        thetaOutputList.append(optForThisRound.copy())
        UList.append(JTest[0] - settings.RISKAVERSION * VTest[0])
        if UList[-1] > optU:
            optU = UList[-1]
            optTheta = optForThisRound.copy()
            if settings.B_PRINTING:
                logger.info("OPT updated. U = %.5f, Theta = %s" % (optU, str(optTheta)))
        # this is not a "legal" way (to find optU, hence optTheta)

    logger.info("Finished computing EPG Policy")                
                
    logger.info("Saving %s Policy" % "EPG")
    fileName ="./seed{}/{}_{}_{}.npy".format(seedList[0],
                                             "EPG"  + ("_TU" if settings.B_TRUEGRADIENT else ''), 
                                      settings.ENV_NAME,
                                      settings.ALGO_NAME)
    np.save(fileName, optTheta)
    
    if trainingPerformance:
        fileName ="./seed{}/{}_{}_{}_Performance.npy".format(seedList[0],
                                                             "EPG" + ("_TU" if settings.B_TRUEGRADIENT else ''), 
                                                      settings.ENV_NAME,
                                                      settings.ALGO_NAME)
        np.save(fileName, trainingPerformance)
    
    logger.info("%s Policy Saved at %s ." % ("EPG", fileName))

    return MakeEPGTestPolicy(optTheta), "EPG", trainingPerformance