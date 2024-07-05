import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import settings



def MakeDetPolicy(actionList):
    def policyOutput(time, state):
        try:
            index = settings.GetTimeStateIndex(time, state)
        except ValueError:
            return np.random.choice(actionList)
        return actionList[index]
    return policyOutput

def MakeStochasticExplorationPolicy(actionList, explorationRate = 0.05):
    def policyOutput(time, state):
        index = settings.GetTimeStateIndex(time, state)
        output = np.random.choice([actionList[index], 
                                 np.random.choice([i for i in range(settings.ACTIONSPACESIZE) if i != actionList[index]])], 
                                p = [1 - explorationRate, explorationRate]) 
        return output
    return policyOutput

def GetEPGFeature(state):
    bTime = settings.FEATURE_ENG[0]       #False
    bConstant = settings.FEATURE_ENG[1]   #True
    bQuadratic = settings.FEATURE_ENG[2]  #False
    bTabular = settings.FEATURE_ENG[3]    #True
    
    if not bTime:
        preFeature = state[:-1] 
    else:
        preFeature = state.copy()
    preFeature = np.concatenate([state, ([1.0] if bConstant else [])])

    if bQuadratic:
        output = []
        preFeature /= np.sum(preFeature**2)**0.5 
        for i in range(len(preFeature)):
            for j in preFeature[i:]:
                output.append(preFeature[i]*j)

        return np.array(output)
    elif bTabular:
        # tabular
        if settings.ENV_TYPE == "PM":
            stateIndex = settings.GetTimeStateIndex(int(state[-1]), state)
        else:
            stateIndex = settings.GetTimeStateIndex(round(5- int(state[1])/2), state)
            
        output = np.zeros(settings.TIMESTATESPACESIZE)
        output[stateIndex] = 1
        return output
    else:
        return preFeature


    #return state #np.concatenate([state, [1.0]])

def MakeEPGTestPolicy(theta):

    def EPGTestingPolicy(time, state):
        feature = GetEPGFeature(state)
        epsilon = settings.EPSILON
        p = epsilon + (1 - 2 * epsilon)/(1+np.exp(-np.dot(theta.copy(), feature)))
        return np.array([1-p, p]).flatten()
    
    def OEEPGTestingPolicy(time, state, debug = False):
        # return the probability of each option
        
        feature = GetEPGFeature(state)
        thetaCopy = theta.copy()
        if len(feature) == 1:
            a = -thetaCopy * feature
        else:
            a = -thetaCopy @ feature
        

        a = np.exp(a)        
        a = a/np.sum(a)

        if np.sum(a[:-1]) <= 1:
            a[-1] = 1 - np.sum(a[:-1])
        else:
            a[-1] = 0.
            ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
            a[ind] -= 1 - np.sum(a[:-1])
        
        #[invest 0, invest 0.1,invest 0.2,invest 0.3,invest 0.4,invest 0.5,invest 0.6, invest 0.7, invest 0.8,invest 0.9, invest 1]
        return a.flatten()
    
    
    if settings.ENV_TYPE == "PM":
        return EPGTestingPolicy
    elif settings.ENV_TYPE == "OptEx":
        return OEEPGTestingPolicy

def SampleTrajectoryNoAgent(env, policy, bStochasticPolicy = False):
    # for cumulative SPERL
    env.reset()
    state = env.getState()
    traj = []
    time = 0
    #try:
    while True:
        if bStochasticPolicy:
            action = np.random.choice(range(settings.ACTIONSPACESIZE), p = policy(time, state))
        else:
            action = policy(time, state)
        newObs, reward, done, info = env.step(action)
        traj.append((state, action, reward))
        time += 1
        state = np.copy(newObs)
        if done:
            break
    #except:
    #    for i in traj:
    #        print(i)
    #    print(state)
    #    raise(ValueError)
    traj.append((state, 0, 0))
    return traj

def SampleTrajectory(env, agent):
    # for EPG
    env.reset()
    state = env.getState()
    traj = []
    
    while True:
        action = agent.DecideAction(state)
        newObs, reward, done, info = env.step(action)
        traj.append((state, action, reward))
        state = np.copy(newObs)
        if done:
            break
    return traj


def ComputeNablaBySimulation(eps, env, agent, trueJ, bDebug = False):
    # compute nabla_J, nabla_V
    # nabla_J = mean[R_0_T-1 * nablalog(P(x_traj))]
    # nabla_J = mean[R_0_T-1 * sum(nabla log(P_action(u)))]
    rz, r2z = [],[]
    trajectories = []
    n = eps
    countValueError = 0
    while eps >0:
        tempRZ, tempR2Z = [],[]
        n = min(eps, 3000)
        for i in range(n):
            #try:
            traj = SampleTrajectory(env, agent).copy()
            r_total = 0
            z = 0
            for state, action, reward in traj:
                r_total += reward
                z += agent.zUpdate(agent.theta, state, action)
            tempRZ.append(r_total * z)
            tempR2Z.append(r_total**2 * z)
        rz.append(np.mean(tempRZ, axis = 0))
        r2z.append(np.mean(tempR2Z, axis = 0))
        eps -= n
    if bDebug:
        print(np.mean(np.std(r2z, axis = 0)), np.mean(np.std(rz, axis = 0)))
    return np.mean(rz, axis = 0), np.mean(r2z, axis = 0) - 2 * np.mean(rz, axis = 0) * trueJ
    
# Train or Test for a given number of eps
def MCTrain(eps, env, agent, bTrain = True, debug = False, report = True):
    if eps == 0:
        return -12332,-12332,-12332,-12332
    # things being tracked
    finalRewards = [0] * eps
    finalWealths = [0] * eps
    
    for i in range(eps):
        trajectory = SampleTrajectory(env, agent)
        finalRewards[i],_,_ = agent.UpdateEps(trajectory, debug = debug, bTrain = bTrain)
        """ 
        if np.isnan(finalRewards[i]):
            print("===============")
            print("Total reward is nan, printing trajectory:")
            for step in trajectory:
                print(step)
            print("===============")
        """ 
        finalWealths[i] = env.GetFinancialPerformance()
    
    # report wealth(not reward)
    estMean_Wealth = np.mean(finalWealths)
    estVar_Wealth = np.std(finalWealths)**2
    
    # report rewards (not wealth)
    estMean = np.mean(finalRewards)
    estVar = np.std(finalRewards)**2
    
    if report:
        print("Mean = %f" %estMean)
        print("Var= %f" %estVar)
        print("============")
    
    return estMean, estVar, estMean_Wealth, estVar_Wealth

def PlotTrainingProgress(meanList, varList, thetaList, wealthList, wealthVar, b, thetaShape, bLegend = False):
    fig = plt.figure(figsize=(7, 10), constrained_layout=True)
    spec = fig.add_gridspec(4, 2)

    ax0 = fig.add_subplot(spec[0, :])

    ax10 = fig.add_subplot(spec[1, 0])

    ax11 = fig.add_subplot(spec[1, 1])

    ax20 = fig.add_subplot(spec[2, 0])

    ax21 = fig.add_subplot(spec[2, 1])

    ax3 = fig.add_subplot(spec[3, :])
    
    #fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=False)

    ax10.plot(meanList)
    ax11.plot(varList)
    ax20.plot(wealthList)
    ax21.plot(wealthVar)
    ax3.plot(np.array(meanList) - settings.RISKAVERSION*np.array(varList))
    
    thetaList = np.array(thetaList).T
    colorList = [clr.hsv_to_rgb(i) for i in [(j/thetaShape[0], 1, 1) for j in range(thetaShape[0])]]
    lsList = ['-', '--', ':']
    counter = 0
    for i in thetaList:
        ax0.plot(i, color = colorList[counter // thetaShape[1]], ls = lsList[counter % thetaShape[1] % len(lsList)], label = counter)
        counter += 1
    
    ax10.set_ylabel('Mean')
    ax11.set_ylabel('Var')
    ax20.set_ylabel('Expected Wealth')
    ax21.set_ylabel('Var (Wealth)')
    ax3.set_ylabel('Utiity')
    if bLegend:
        ax0.legend(loc='upper left')
    #ax0.set_ylim([-50, 100])
    fig.suptitle('Tracking Mean & Varianc during training with b = %.7f' % b)
    print("Mean: %.7f, Var: %.7f" % (meanList[-1], varList[-1]))
    plt.show()