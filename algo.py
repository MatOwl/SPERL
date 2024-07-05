import numpy as np
import pandas as pd

import settings
import Agents
from tools import *
from OptExTree import *
from environments import *

import logging

logger = logging.getLogger('root')


def NextActionList(currentAL):
    done = False
    lastChangedIndex = 0
    
    currentAL[0] +=1
    for i,j in enumerate(currentAL):        
        if j == 2:
            currentAL[i] = 0
            if settings.TIMESTATESPACE[i][0] == settings.HORIZON:
                done = True
                break
            currentAL[i+1] += 1
        else:
            lastChangedIndex= i
            break
            
    return done, lastChangedIndex


def ComputePreCommitmentOptPolicy(envStr = "PM"):
    logger.info("Start computing PreCom Opt Policy")
    
    optPolicyPreCom = lambda time,state:0
    optU = 0
    actionList = [0] * settings.TIMESTATESPACESIZE
    optAL = actionList.copy()
    tree = OptExTree(optPolicyPreCom,
                     env = settings.ENV,
                     initialStates = settings.INITIAL_STATES,
                     bStochasticPolicy = False,
                     bComputeMoment = True,
                     bComputeChild = True
                     )
    bDone = False
    lastChangedIndex = 0
    maxIndex = 0

    while not bDone:
        policyTemp = MakeDetPolicy(actionList.copy())
        tree.policy = policyTemp
        ## Update J and M according to change in policy
        for i in range(lastChangedIndex):
            key = settings.TIMESTATESPACE[i]
            tree.stateTimeSpaceDict[key].J = np.array([None,None])
            tree.stateTimeSpaceDict[key].M = np.array([None,None])
        tree.ComputeMoment(False)

        

        ## evaulate
        U = 0
        for root in tree.roots:
            action = policyTemp(0, root.GetState())
            J = root.J[action]
            M = root.M[action]
            V = M - J **2
            U += (J - settings.RISKAVERSION * V)
        U /= len(tree.roots)
        """
        if maxIndex == 17 and U > round(optU, 4):
            logger.info("Trying policy %s" % actionList[:settings.TIMEDIVIDER[-1]])  
            logger.info("U = %.7f (%.7f)" % (U, optU))
        """
        
        if optU < U:
            optU = U
            optPolicyPreCom = policyTemp
            optAL = actionList.copy()
            logger.info("Change happens at %d. Best U = %.5f, Action List = %s." % (lastChangedIndex, 
                                                                                    optU, 
                                                                                    optAL[:settings.TIMEDIVIDER[-1]]))
            
        bDone, lastChangedIndex = NextActionList(actionList)
        if lastChangedIndex > maxIndex:
            maxIndex = lastChangedIndex
            if settings.TIMESTATESPACE[maxIndex][0] == settings.HORIZON:
                break
            logger.info("Last Changed Index is %d at Height %d." % (lastChangedIndex, 
                                                                    settings.TIMESTATESPACE[lastChangedIndex][0]))
    logger.info("Finished computing PreCom Opt Policy")
    
    logger.info("Saving %s Policy" % "PreCom")
    fileName ="./{}_{}.npy".format("PreCom", 
                                   settings.ENV_NAME) 
    np.save(fileName, optAL)
    logger.info("%s Policy Saved at %s ." % ("PreCom", fileName))

    
    return MakeDetPolicy(optAL), "PreCom", []


def ComputeSPEOptPolicy(envStr = "PM"):
    logger.info("Start computing SPE")
    
    optPolicy = lambda time,state:1
    optU = -100
    actionList = [0] * settings.TIMESTATESPACESIZE
    tree = OptExTree(optPolicy,
                     env = settings.ENV,
                     initialStates = settings.INITIAL_STATES,
                     bStochasticPolicy = False,
                     bComputeMoment = True,
                     bComputeChild = True
                     )
        
    currentHeight = settings.HORIZON
    logger.info("Comuting SPE at Height = %d." % currentHeight)

    for i,key in enumerate(settings.TIMESTATESPACE[::-1]):
        
        if key[0] == settings.HORIZON:
            continue

        if key[0] != currentHeight:
            currentHeight = key[0]
            optPolicy = MakeDetPolicy(actionList.copy())
            tree = OptExTree(optPolicy,
                             env = settings.ENV,
                             initialStates = settings.INITIAL_STATES,
                             bStochasticPolicy = False,
                             bComputeMoment = True,
                             bComputeChild = True
                             )
            #print(actionList)
            logger.info("Comuting SPE at Height = %d." % currentHeight)
        
        node = tree.stateTimeSpaceDict[key]
        V = node.M - node.J **2
        U = node.J - settings.RISKAVERSION * V
        actionList[settings.TIMESTATESPACESIZE -1 - i] =  np.argmax(U) #int(U[0] < U[1])

    optPolicy = MakeDetPolicy(actionList.copy())
    SPE_AL = np.array(actionList.copy())
    #print(actionList)
    logger.info("Finish computing SPE")
    
    logger.info("Saving %s Policy" % "SPE")
    fileName ="./{}_{}.npy".format("SPE", 
                                   settings.ENV_NAME) 
    np.save(fileName, SPE_AL)
    logger.info("%s Policy Saved at %s ." % ("SPE", fileName))
    
    return optPolicy, "SPE", []


#==============================
#    CumSPERL
#==============================
def PolicyEvlTDZero(eps, 
           env, 
           policyChosen, 
           bStochasticPolicy, 
           JMTable,
           selectedTime = -1,   # (settings.HORIZON - 1) to 0 [reversed(range(settings.HORIZON))]
           trajList = [], 
           reuseNumber = 0,
           stepSize = 0.1
           ):
    
    #=====================================
    # complete the training data set
    gap = eps - len(trajList)
    if gap > 0:
        trajList = trajList
        for k in range(gap):
            trajSamp = SampleTrajectoryNoAgent(env, policyChosen, bStochasticPolicy = bStochasticPolicy)
            trajList.append(trajSamp)
    #=====================================
    
    
    #stateTimeFrequency = np.zeros(settings.LEN_FEATURE)

    ## actual simulation
    for k in range(eps):
        errorArray = []   # for debug

        for traj in trajList[max(0, k - reuseNumber): k+1]:  # [trajSamp]: 
            """
            print("PolicyEvlTDZero: considering the following Traj")
            for i in traj:
                print(i)
            print("PolicyEvlTDZero: considering the following state")
            print(traj[selectedTime])
            """
            lpr, rt, rh = traj[selectedTime][0]
            action = traj[selectedTime][1]
            feature = ((JMTable['logPriceRatio'] == lpr) &
                       (JMTable['rHold'] == rh) &
                       (JMTable['rTime'] == rt) &
                       (JMTable['action'] == action)
                      )
            
            nextTime = selectedTime + 1

            # stateTimeFrequency += feature # debug    Rewrite this for new version

            TDTermJ = traj[selectedTime][2] 
            TDTermM = traj[selectedTime][2]**2

            try:
                temp = JMTable[feature].iloc[0]
            except IndexError:
                df2 = {'logPriceRatio': lpr, 
                       'rHold': rh, 
                       'rTime': rt,
                       'action':action,
                       'J': 0,
                       'M': 0,
                       'JAdj': 0,
                       'MAdj': 0,
                       'Q' : 1   # encourage exploring
                      }
                JMTable = pd.concat([JMTable, pd.DataFrame(df2, index = np.arange(1))], ignore_index = True)
                JMTable.reset_index()
                feature = ((JMTable['logPriceRatio'] == lpr) &
                       (JMTable['rHold'] == rh) &
                       (JMTable['rTime'] == rt) &
                       (JMTable['action'] == action)
                      )
            currentEstJ = JMTable[feature].iloc[0].at['J']
            currentEstM = JMTable[feature].iloc[0].at['M']
            
            if nextTime == len(traj)-1:
                # simplified as J and M for last state are always 0
                TDTermJ += 0 - currentEstJ
                TDTermM += 0 - currentEstM
            else:                    
                """
                if bStochasticPolicy:
                    nextActionP = policyChosen(nextTime, traj[nextTime][0])

                    nextFeature_0 = settings.ComputeFeature(nextTime, traj[nextTime][0], 0)                     
                    nextFeature_1 = settings.ComputeFeature(nextTime, traj[nextTime][0], 1)                     

                    TDTermJ += nextActionP[0] * (settings.DISCOUNTING * nextFeature_0 - feature) @ paraVectorJ
                    TDTermJ += nextActionP[1] * (settings.DISCOUNTING * nextFeature_1 - feature) @ paraVectorJ

                    TDTermM += nextActionP[0] * (2 * settings.DISCOUNTING * traj[selectedTime][2] * nextFeature_0 @ paraVectorJ + 
                                                 (settings.DISCOUNTING**2 * nextFeature_0 - feature) @ paraVectorM)
                    TDTermM += nextActionP[1] * (2 * settings.DISCOUNTING * traj[selectedTime][2] * nextFeature_1 @ paraVectorJ + 
                                                 (settings.DISCOUNTING**2 * nextFeature_1 - feature) @ paraVectorM)

                else:
                """
                
                nextAction = policyChosen(nextTime, traj[nextTime][0])
                nextFeature = ((JMTable['logPriceRatio'] == traj[nextTime][0][0]) &
                               (JMTable['rHold'] == traj[nextTime][0][2]) &
                               (JMTable['rTime'] == traj[nextTime][0][1]) &
                               (JMTable['action'] == nextAction)
                              )
                try:
                    currentEstNextJ = JMTable[nextFeature].iloc[0].at['J']
                    currentEstNextM = JMTable[nextFeature].iloc[0].at['M']
                except IndexError:
                    print(nextTime)
                    print(traj)
                    print(traj[nextTime])
                    print(JMTable.loc[(JMTable['logPriceRatio'] == traj[nextTime][0][0]) &
                               (JMTable['rHold'] == traj[nextTime][0][2]) &
                               (JMTable['rTime'] == traj[nextTime][0][1])])
                TDTermJ += settings.DISCOUNTING * currentEstNextJ - currentEstJ
                TDTermM += (2 * settings.DISCOUNTING * traj[selectedTime][2] * currentEstNextJ + 
                            settings.DISCOUNTING**2 * currentEstNextM - currentEstM
                           )
            JMTable.loc[feature, 'JAdj'] += TDTermJ
            JMTable.loc[feature, 'MAdj'] += TDTermM
        ## update parameters
        JMTable['J'] += JMTable["JAdj"] * stepSize
        JMTable['M'] += JMTable["MAdj"] * stepSize
        JMTable["JAdj"] = 0
        JMTable["MAdj"] = 0
        
    return JMTable

class OEAgent():
    def __init__(self, df):
        self.df = df
        self.exploreProb = 0
        
        
    def SetExploreProb(self, prob):
        self.exploreProb = prob
        
        
    def GetAction(self, time, state, bExplore = False):

        lpr, rt, rh = state
        try:
            action = self.df[(self.df['logPriceRatio'] == lpr) &
                             (self.df['rHold'] == rh) &
                             (self.df['rTime'] == rt)].iloc[0].at['policyAction']
            if bExplore and np.random.rand() < self.exploreProb:
                action = (action + np.random.choice([1,2,3,4,5,-1,-2,-3,-4,-5])) % 11
            if rt == 2:
                action = 10
        except IndexError:
            # hard-coded
            action = 10 if rt == 2 else np.random.choice(range(11))
            tempDf = {'logPriceRatio': lpr, 
                       'rHold': rh, 
                       'rTime': rt,
                       'policyAction':action
                      }
            self.df = pd.concat([self.df.copy(), pd.DataFrame(tempDf, index = np.arange(1))], ignore_index = True)
            self.df.reset_index()
        return action 
    
    def ExportActionList(self, timeStateSpace):
        output = [0] * len(timeStateSpace)
        for i in range(len(timeStateSpace)):
            time, state = timeStateSpace[i]
            state = eval("["+state+"]")
            output[i] = int(self.GetAction(time, state, bExplore = False))
        return output
    
    def UpdateBestAction(self, bestRecords):
        self.df = self.df.merge(bestRecords, how='left', on=['logPriceRatio', 'rTime', 'rHold'])
        #print(self.df.loc[pd.notna(self.df['action']) & (self.df['policyAction'] !=self.df['action'])].shape)
        self.df.loc[pd.notna(self.df['action']), 'policyAction'] = self.df.loc[:, 'action']
        self.df = self.df.drop(columns=['action'])
    
    def Summary(self):
        print(self.df.groupby(['rTime','policyAction']).count())
        
    def LoadPolicy(self, fileName):
        self.df = pd.read_csv(fileName)  
        
    def SavePolicy(self, fileName):
        self.df.to_csv(fileName,index=False)      
        

def ComputeSPERLPolicy(envStr = "PM", agent = None, table = False, initialSS = 0.1, 
                       maxRound = 1500, 
                       bInspect = False):   
    # mark(to be updated) ========================
    env = settings.ENV
    stepSize_J_M = initialSS
    ssFactor = 1.
    randomSeed = settings.RANDOM_SEED
    np.random.seed(randomSeed)
    if settings.FORCE_UPDATE:
        trainingPerformance = list(np.load("./seed{}/{}_{}_Performance.npy".format(randomSeed,
                                                          "SPERL", 
                                                   settings.ENV_NAME) ))
        trainingEpsSPERL = 4000#1000
        exploreRate = 0.0
        exploreRateDecay = 0.999
    else:
        trainingPerformance = []
        trainingEpsSPERL = 4000#1000
        exploreRate = 0.1
        exploreRateDecay = 0.999

    
    if type(table) != bool:
        JMTable = table
    else:
        try:
            fileName = "./seed{}/{}_{}.csv".format(randomSeed,
                                                   "SPERL", 
                                                   settings.ENV_NAME)
            JMTable = pd.read_csv(fileName, index_col = 0 )
            logger.info("JMV table loaded")
        except FileNotFoundError:
            logger.info("JMV table initialised")
            
            JMTable = pd.DataFrame(
                                {
                                    "logPriceRatio": [0],
                                    "rHold": [10],
                                    "rTime": [10],
                                    "action": [0],
                                    "J": [10],
                                    "M": [0],
                                    "JAdj":[0],
                                    "MAdj":[0],
                                    "Q":[0]
                                }
                            )

    if agent == None:
        try:
            fileName = "./seed{}/{}_{}_Policy.csv".format(randomSeed,"SPERL", settings.ENV_NAME)
            PolicyTable = pd.read_csv(fileName)
            logger.info("Policy table loaded")
        except FileNotFoundError:
            logger.info("JMV table initialised")
            PolicyTable = pd.DataFrame({"logPriceRatio": [0],
                                        "rHold": [10],
                                        "rTime": [10],
                                        "policyAction": [0]})

        oeAgent = OEAgent(PolicyTable)
        oeAgent.SetExploreProb(exploreRate)
        print("New agent created")
        
    else:
        oeAgent = agent
        print("Agent loaded")
    
    counter = 0
    actionListList = []
        
    while (counter < (50 if settings.FORCE_UPDATE else maxRound)):
        #logger.info("Training Round %d" % (counter+1))
            
        policyExplore = lambda time,state: oeAgent.GetAction(time, state, bExplore = True)
        
        trajList = [[]] * trainingEpsSPERL
        for k in range(trainingEpsSPERL):
            try:
                trajSamp = SampleTrajectoryNoAgent(env, policyExplore, bStochasticPolicy = False)
                trajList[k] = trajSamp
            except ValueError:
                raise ValueError("Not found In List again at %d" % k)
                #logger.warning("Not found In List again at %d" % k)
        
        for time in range(settings.HORIZON-1,-1,-1):   # 3,2,1,0
            policyChosen = lambda t,state: oeAgent.GetAction(t, state, bExplore = False)
            JMTable = PolicyEvlTDZero(trainingEpsSPERL, 
                                         env, 
                                         policyChosen, 
                                         False, 
                                         JMTable,
                                         selectedTime = time,
                                         trajList = trajList,
                                         stepSize = stepSize_J_M
                                        )

            # improvement: 
            ## from 4,3,2,1,0 to 2,4,6,8,10
            rtNow = (settings.HORIZON - time) * (10//settings.HORIZON)
            JMTable.loc[JMTable['rTime'] == rtNow , "Q"] = (JMTable.loc[JMTable['rTime'] == rtNow , "J"] - 
                                                            settings.RISKAVERSION*(JMTable.loc[JMTable['rTime'] == rtNow , "M"] - 
                                                                                   JMTable.loc[JMTable['rTime'] == rtNow , "J"]**2))
            thisTimeTable = JMTable[JMTable['rTime'] == rtNow]
            """
            if rtNow == 10 and bInspect:
                print(thisTimeTable.loc[:,['action', 'J', 'M', "Q"]])
            """
            ## group those at the same state
            maxIds = thisTimeTable.groupby(['logPriceRatio', 'rTime', 'rHold']).idxmax().loc[:,'Q']
            bestRecords = JMTable.loc[maxIds,['logPriceRatio', 'rTime', 'rHold', 'action']]
            oeAgent.UpdateBestAction(bestRecords)
        print(JMTable.loc[:, "Q"].mean(), JMTable.shape)
        """
        if bInspect:
            print(oeAgent.df.describe())
        """
        
        """
        eps = 10000
        cumulatedRewardList = np.array([0.0] * eps)
        #oeAgent.Summary()
        policyChosen = lambda t,state: oeAgent.GetAction(t, state, bExplore = False)
        for k in range(eps):
            cumulatedReward = 0
            trajSamp = SampleTrajectoryNoAgent(env, policyChosen, bStochasticPolicy = False)
            for i in trajSamp:
                cumulatedReward += i[2]
            cumulatedRewardList[k] = cumulatedReward

        mean = np.mean(cumulatedRewardList)
        var = np.std(cumulatedRewardList)**2
        """
        counter += 1
        # use the true value to test performance :
        if counter % ((50 if settings.FORCE_UPDATE else maxRound)//3) == 0:
            stepSize_J_M *= 0.3
            stepSize_J_M = max(stepSize_J_M, 0.05)
            
        if counter % 5 == 0:
            exploreRate *= exploreRateDecay
            oeAgent.SetExploreProb(exploreRate)
            #logger.info("Testing Starts")
            
            tempPolicy = lambda t,state: oeAgent.GetAction(t, state, bExplore = False)
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
            trainingPerformance.append(JTest[-1] - settings.RISKAVERSION * VTest[-1])
            if trainingPerformance:
                fileName ="./seed{}/{}_{}_Performance.npy".format(randomSeed,
                                                                  "SPERL", 
                                                           settings.ENV_NAME) 
                np.save(fileName, trainingPerformance)
        
        outputAL = oeAgent.ExportActionList(settings.TIMESTATESPACE)
        if counter % 5 == 0:
            #logger.info("Saving CSV")
            oeAgent.SavePolicy("./seed{}/{}_{}_Policy.csv".format(randomSeed,
                                                                  "SPERL", 
                                                                  settings.ENV_NAME) )
            JMTable.to_csv("./seed{}/{}_{}.csv".format(randomSeed,
                                                       "SPERL", 
                                                       settings.ENV_NAME) )  
            logger.info("Saving %s Policy" % "SPERL")
            fileName ="./seed{}/{}_{}_{}.npy".format(randomSeed,
                                                     "SPERL", 
                                     settings.ENV_NAME,
                                      settings.ALGO_NAME)
            np.save(fileName, outputAL)
    

    logger.info("Finished computing SPERL Policy")
    if counter == (50 if settings.FORCE_UPDATE else maxRound):
        logger.warning("SPERL policy does not converge.")
    
    outputAL = oeAgent.ExportActionList(settings.TIMESTATESPACE)
    logger.info("Saving %s Policy" % "SPERL")
    fileName ="./seed{}/{}_{}_{}.npy".format(randomSeed,
                                             "SPERL", 
                                     settings.ENV_NAME,
                                      settings.ALGO_NAME)
    np.save(fileName, outputAL)
    
    if trainingPerformance:
        fileName ="./seed{}/{}_{}_Performance.npy".format(randomSeed,
                                                          "SPERL", 
                                                   settings.ENV_NAME) 
        np.save(fileName, trainingPerformance)
    
    logger.info("Saving CSV")
    JMTable.to_csv("./seed{}/{}_{}.csv".format(randomSeed,
                                               "SPERL", 
                               settings.ENV_NAME) )  
    logger.info("%s Policy Saved at %s ." % ("SPERL", fileName))
    
    #return oeAgent, JMTable
    return MakeDetPolicy(outputAL), "SPERL", trainingPerformance
    
    
def ComputeEPGPolicy(envStr = "PM"):
    # ============== Main =======================
    # Train and test combined and repeat
    def Main(agent, env, numOfRound = 100, trainEps = 100, testEps = 0, seed = 0):
        meanList = []
        varList = []
        thetaList = []
        wealthList = []
        wealthVarList = []
        trainingPerformance = []
        
        #logger.info("EPG Main Starts")
        
        ss = 100   # 1000 -> 500000
        n_sim = 1000 if settings.bSPERL_TABULAR else 10000
        trueU = -1000
        
        if settings.EXTEND_TO_150:
            ss = 3200
            n_sim = 32000
            
        for i in range(numOfRound):       
            #logger.info("EPG Main Round %d Starts" % (i))
            # EPG method
            # MCTrain(trainEps, env, agent, bTrain = True, debug = False, report = False)
            print("ss Now = %d" % (ss))
            if settings.B_TRUEGRADIENT:      
                # true update by nabla
                tempPolicy = MakeEPGTestPolicy(agent.theta)
                tempTree = OptExTree(tempPolicy,
                                     env = settings.ENV,
                                     initialStates = settings.INITIAL_STATES,
                                     bStochasticPolicy = True,
                                     bComputeMoment = True
                                     )
                trueJ = 0
                trueUTemp = trueU
                trueU = 0

                for rootNode in tempTree.roots:
                    actionP = tempPolicy(0, rootNode.GetState())
                    trueJ += (rootNode.J @ actionP) / len(tempTree.roots)
                    trueU += (rootNode.J - settings.RISKAVERSION * 
                              (rootNode.M - rootNode.J**2))@ actionP / len(tempTree.roots)

                logger.info(str(trueU))

                if trueU - trueUTemp < 0.00001:
                    if settings.EXTEND_TO_150:
                        ss = ss//2
                    else:
                        ss = min(ss * 2, 30000)
                        n_sim = min(n_sim * 2, 200000)
                        if ss > (1000 if settings.FORCE_UPDATE else 12800):
                            break

                #logger.info("EPG Main Round %d Compute Nabla" % (i))

                # print("===============1000===================")
                # for repeat in range(10):
                #     A,B = ComputeNablaBySimulation(1000, env, agent, trueJ)
                #     print(np.mean(A),np.mean(B))
                # print("===============10000===================")
                # for repeat in range(10):
                #     A,B = ComputeNablaBySimulation(10000, env, agent, trueJ)
                #     print(np.mean(A),np.mean(B))
                # print("===============100000===================")
                # for repeat in range(10):
                #     A,B = ComputeNablaBySimulation(100000, env, agent, trueJ)
                #     print(np.mean(A),np.mean(B))


                nabla_J, nabla_V = ComputeNablaBySimulation(n_sim, env, agent, trueJ)
                agent.SetStepSize(0.005, ss if settings.bSPERL_TABULAR else 0.1, 0)

                agent.TrueUpdate(nabla_J*(1), nabla_V*(1))
                trainingPerformance.append(trueU)
            else:
                # for Approximated gradient method
                agent.SetStepSize(0.005, 0.05, 0)
                tempPolicy = MakeEPGTestPolicy(agent.theta)
                tempTree = OptExTree(tempPolicy,
                                     env = settings.ENV,
                                     initialStates = settings.INITIAL_STATES,
                                     bStochasticPolicy = True,
                                     bComputeMoment = True
                                     )
                trueJ = 0
                trueUTemp = trueU
                trueU = 0

                
                for rootNode in tempTree.roots:
                    actionP = tempPolicy(0, rootNode.GetState())
                    trueU += (rootNode.J - settings.RISKAVERSION * (rootNode.M - rootNode.J**2))@ actionP / len(tempTree.roots)
                logger.info(str(trueU))
                trainingPerformance.append(trueU)
                
                MCTrain(trainEps * 2000, env, agent, bTrain = True, debug = False, report = False)


            
            
            
#             testMean, testVar, wealthMean, wealthVariance = MCTrain(testEps, 
#                                                                     env, 
#                                                                     agent, 
#                                                                     bTrain = False, 
#                                                                     debug = False, 
#                                                                     report = False)
            
#             meanList.append(testMean)
#             varList.append(testVar)
#             wealthList.append(wealthMean)
#             wealthVarList.append(wealthVariance)
            thetaList.append(agent.theta.flatten())
            fileName ="./seed{}/{}_{}_{}.npy".format(settings.RANDOM_SEED, 
                                                     "EPG"  + ("_TU" if settings.B_TRUEGRADIENT else ''), 
                                  settings.ENV_NAME,
                                  settings.ALGO_NAME)
            np.save(fileName, agent.theta)
            

        return meanList, varList, thetaList, wealthList, wealthVarList, trainingPerformance

    def MakeZUpdate(actionSpace, thetaShape, epsilon = 0.05):
        def ZUpdate(theta, state, action):
            expTerm = np.exp(-np.dot(theta, state))
            if action == 1:
                nominator = epsilon + (1 - 2* epsilon) / (1+expTerm)
                tempReal = (1 - 2* epsilon) * expTerm / (1+expTerm)**2/ nominator
            else:
                nominator = 1 - epsilon - (1 - 2* epsilon) / (1+expTerm)
                tempReal = -(1 - 2* epsilon) * expTerm/ (1+expTerm)**2/ nominator
            temp = tempReal * state
            return temp
        return ZUpdate
    
    def MakeEPGPolicy(epsilon):
        def policy(theta, state, debug = False):
            # return the probability of each option
            p = epsilon + (1 - 2 * epsilon)/(1+np.exp(-np.dot(theta, state)))[0]
            # dot product negative, epsilon
            # dot product positive, 1 - epsilon
            return np.array([1-p, p])

        return policy
    
    
    def MakeOEEPGPolicy(epsilon = settings.EPSILON):
        def policy(theta, state, debug = False):
            # return the probability of each option
            feature = GetEPGFeature(state)
            
            if len(feature) == 1:
                a = -theta * feature
            else:
                a = -theta @ feature

            p = np.exp(a)
            p = (p + epsilon)/(np.sum(p) + epsilon * theta.shape[0])
            
            if np.sum(p[:-1]) <= 1:
                p[-1] = 1 - np.sum(p[:-1])
            else:
                p[-1] = 0.
                ind = np.unravel_index(np.argmax(p, axis=None), p.shape)
                p[ind] -= 1 - np.sum(p[:-1])
                
            if any(p < 0):
                print(p)
            return p

        return policy

    def MakeOEZUpdate(actionSpace, thetaShape, epsilon = settings.EPSILON):
        def ZUpdate(theta, state, action):
            feature = GetEPGFeature(state)
            
            if len(feature) == 1:
                a = -theta * feature
            else:
                a = -theta @ feature

            expTerm = np.exp(a)
            E = np.sum(expTerm) + epsilon * theta.shape[0]
            p = (expTerm + epsilon)/E
            
            actionIndex = actionSpace.index(action)
            output = np.zeros(theta.shape)
            for i in range(theta.shape[0]):
                if i == actionIndex:
                    output[i] = (expTerm[actionIndex] * (expTerm[actionIndex] + epsilon - E))/(E * E) * feature
                else:
                    output[i] = (expTerm[i] * (expTerm[actionIndex] + epsilon))/(E * E) * feature

            output = output / p[actionIndex]
            return output
        return ZUpdate
    
    """
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

            #[invest 0, invest 0.1,invest 0.2,invest 0.3,invest 0.4,
              invest 0.5,invest 0.6, invest 0.7, invest 0.8,invest 0.9, invest 1]
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
            
            # if len(feature) == 1:
            #     sums = np.sum(theta) * feature
            # else:
            #     sums = np.sum(np.dot(theta, feature))
            # temp = np.array([-1 * feature / sums] * thetaShape[0]).reshape(thetaShape)
            # temp[actionIndex] += np.copy(feature)
            return temp
        return ZUpdate
    """
            
    trainEps = 1
    testEps = 1
    trainRound = 100
    trainingPerformance = []
    env = settings.ENV
    
    # agent
    if settings.ENV_TYPE == "PM":
        thetaSize = [1 ,2 + settings.ILLIQUIDPERIOD]
        actionSpace = [0,1]
        policy = MakeEPGPolicy(0.05)
        zUpdateFn = MakeZUpdate(actionSpace, thetaSize)
    elif settings.ENV_TYPE == "OptEx":
        thetaSize = [settings.ACTIONSPACESIZE, len(GetEPGFeature(env.getState()))]
        print(thetaSize)
        
        actionSpace = [i for i in range(settings.ACTIONSPACESIZE)]
        policy = MakeOEEPGPolicy()
        zUpdateFn = MakeOEZUpdate(actionSpace, thetaSize)

    env = settings.ENV
    
    # result plotting
    EFvarianceList = []
    EFwealthList = []
    thetaOutputList = []
    UList = []

    g = settings.RISKAVERSION
    seedList = [settings.RANDOM_SEED]#[int(np.random.random()*100)]#[16, 46572,345, 63]#,3545154,34,325,53,75]

    optU = -100
    optTheta = []
    
    for s in seedList:
        logger.info("Training EPG for seed %d" % s)
        np.random.seed(s)
        if settings.FORCE_UPDATE:
            initTheta = np.load("./seed{}/{}_{}_{}.npy".format(seedList[0],
                                                      "EPG"  + ("_TU" if settings.B_TRUEGRADIENT else ''), 
                                      settings.ENV_NAME,
                                      settings.ALGO_NAME)
                               )
            fileName ="./seed{}/{}_{}_{}_Performance.npy".format(seedList[0],
                                                      "EPG"  + ("_TU" if settings.B_TRUEGRADIENT else ''), 
                                      settings.ENV_NAME,
                                      settings.ALGO_NAME)
            try:
                trainingPerformance = list(np.load(fileName))
            except FileNotFoundError:
                trainingPerformance = []
            if settings.EXTEND_TO_150:
                trainRound = 150 - len(trainingPerformance)
                
        else:
            initTheta =  (np.random.rand(*thetaSize) - 0.5)* 1
        # define agent accordingly
        EPGAgent = Agents.EPG(thetaSize, 
                                      actionSpace, 
                                      policy, 
                                      zUpdateFn, 
                                      initTheta = initTheta, 
                                      objectiveType = 1, 
                                      gamma = g)


        # Train and Collect data
        meanList, varList, thetaList, wealth, wealthVar,Performance = Main(EPGAgent, 
                                                                           env,
                                                                           trainRound, 
                                                                           trainEps, 
                                                                           testEps,
                                                                          s)
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
        JTest, VTest, theta_output, wealth, wealthVar,TrueJ = Main(EPGAgent, env, 1, 0, 5000)
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
                                                      "EPG"  + ("_TU" if settings.B_TRUEGRADIENT else ''), 
                                      settings.ENV_NAME,
                                      settings.ALGO_NAME)
        np.save(fileName, trainingPerformance)
    
    logger.info("%s Policy Saved at %s ." % ("EPG", fileName))

    return MakeEPGTestPolicy(optTheta), "EPG", trainingPerformance
