import numpy as np
import settings

class OptExTree():
    def __init__(self, 
                 policy,
                 env, 
                 initialStates,
                 bStochasticPolicy = False,
                 bComputeMoment = False,
                 bProgressBar = True,
                 bComputeChild = True
                 ):
        self.policy = policy
        self.env = env
        self.bStochasticPolicy = bStochasticPolicy
        
        self.stateTimeSpaceDict = {}
        
        self.agent = None
        self.roots = [OptExNode(self, 0, state = state) for state in initialStates]
        for root in self.roots:
            root.UpdateParent(None, (None, 0, 1.0/len(initialStates)))
            
        if bComputeChild:
            self.ComputeChild()
        
        #if bProgressBar:
            #print("All children created!")
        if bComputeMoment:
            self.ComputeMoment(bStochasticPolicy)
    """
    def AddChild(self, childNode):
        key = (childNode.height, str(childNode))
        if key not in self.stateTimeSpaceDict.keys():
            self.stateTimeSpaceDict[key] = childNode
        else:
            childNode.child = self.stateTimeSpaceDict[key].child
    """
    def ComputeChild(self):
        lastLayer = self.roots
        for node in lastLayer:
            self.stateTimeSpaceDict[(0, str(node))] = node
        
        newLayer = []
        layerNext = 0
        while len(lastLayer) != 0:    # change it to while not empty
            layerNext += 1
            for node in lastLayer:
                currentState = node.GetState()
                for options in self.env.optionsList:
                    # ================================
                    # do some thing
                    newState, reward, done = self.env.NextState(currentState, node.height, *options)
                    #print(currentState, options, newState)
                    newReachingProb = self.env.OptionsProb(options)   # only about W, not about action
                    newNode = OptExNode(self, node.height + 1, state = newState) 
                    newNode.UpdateParent(node, (options[0], reward, newReachingProb))
                    # ================================
                    
                    key = (newNode.height, str(newNode))
                    #print("Create: " + str(key))
                    if key not in self.stateTimeSpaceDict.keys():
                        self.stateTimeSpaceDict[key] = newNode
                        #print("In: " + str(key))
                        if not (done or layerNext == self.env.horizon):
                            # if next layer is the last layer, then don't need to consider
                            newLayer.append(newNode)
                        node.child.append(newNode)
                    else:
                        #print("Repeated, use existing node:" + str(self.stateTimeSpaceDict[key]))
                        newNode = self.stateTimeSpaceDict[key]
                        if newNode.UpdateParent(node, (options[0], reward, newReachingProb)):
                            node.child.append(newNode)

                    
                        
            lastLayer = newLayer.copy()
                
            newLayer = []
    
    def ComputeMoment(self, bStochastic):
        # something recursive
        # make use of the final cumulative rewards and probablity along the way
        for node in self.roots:
            node.ComputeMoments(self.policy, bStochasticPolicy = bStochastic)
    
    def ComputeVisitProb(self, bStochastic):
        """
        """
        nodeToBeCompute = [root for root in self.roots]
        for root in nodeToBeCompute:
            # surely, there is no parent for root. 
            # hence no need to take care about the case where 
            #    parent has 2 actions reaching the same child
            root.AddVisitProb(root.parent[None][0][2])
            
        time = 0
        
        while time < settings.HORIZON:
            temp_sumToOne = 0
            tempList = []
            
            tempDoneNode = []
            for node in nodeToBeCompute:                
                if node in tempDoneNode:
                    continue
                else:
                    tempDoneNode.append(node)
                    #temp_sumToOne += node.visitProb   # debug: should sum to 1
                    if bStochastic:
                        actionP = self.policy(time, node.GetState())
                    else:
                        action = self.policy(time, node.GetState())
                        actionP = np.array([action == i for i in range(settings.ACTIONSPACESIZE)], dtype = np.float32)

                    for child in node.child:
                        
                        bReaching = False
                        for info in child.parent[node]:
                            prob = node.visitProb * actionP[info[0]] # prob: probability of reaching child with given policy
                            if prob <= 0:
                                continue
                            else:
                                bReaching = True

                            child.AddVisitProb(prob * info[2])
                        if bReaching:
                            tempList.append(child)

            nodeToBeCompute = tempList.copy()
            time += 1
    def ComputePolicyGradient(self, theta, zFun):
        # theta is an array       
        # only support for one root tree
        output = self.roots[0].ComputePolicyGradientEPG(theta, zFun, None, [], self.policy)
        return output
        
            
            
class OptExNode():
    
    def __init__(self, tree, height, state = []):        
                
        self.height = height
        self.child = []
        self.parent = {} #[(action, reward, prob),()]   # prob use for computing true J, M, U
        
        self.state = state   # np.array
        
        self.visitProb = 0   # use for U weighting and sorting. the P in main
        
        self.J = np.array([None] * settings.ACTIONSPACESIZE)
        self.M = np.array([None] * settings.ACTIONSPACESIZE)
        
        
    def ComputeMoments(self, policy, bStochasticPolicy= False):
                        
        if not self.child:
            self.J = np.array([0] * settings.ACTIONSPACESIZE,dtype = np.float32)
            self.M = np.array([0] * settings.ACTIONSPACESIZE,dtype = np.float32)
            # return
        elif self.J[0] == None:   # compute only if not yet computed
            
            nextStepReward = np.array([0] * settings.ACTIONSPACESIZE,dtype = np.float32)
            nextStepRewardSqr = np.array([0] * settings.ACTIONSPACESIZE,dtype = np.float32)
            for childNode in self.child:
                for info in childNode.parent[self]:
                    actionVector =  np.array([info[0] == i for i in range(settings.ACTIONSPACESIZE)],
                                             dtype = np.float32)
                    nextStepReward += info[1] * info[2] * actionVector
                    nextStepRewardSqr += (info[1] ** 2) * info[2] * actionVector
                        
            self.J = nextStepReward * 1.0
            self.M = nextStepRewardSqr * 1.0
            
            for childNode in self.child:
                childState = childNode.ComputeMoments(policy, bStochasticPolicy)
                for info in childNode.parent[self]:
                    canBeReached = np.array([info[0] == i for i in range(settings.ACTIONSPACESIZE)],
                                             dtype = np.float32)
                    # if child.reaching action == given action (index) then update J,M for that action accordingly

                    #self.height == self.tree.horizon - 1
                    if len(childNode.child) == 0:
                        actionWeighChild = np.array([1 / settings.ACTIONSPACESIZE] * settings.ACTIONSPACESIZE,dtype = np.float32) 
                        # dummy. since J, M for last node are 0 for all action
                    else:
                        if bStochasticPolicy:
                            actionWeighChild = policy(self.height+1, childState)   # an array of probabilities for each action
                        else:
                            actionChild = policy(self.height+1, childState)
                            actionWeighChild = np.array([actionChild == i for i in range(settings.ACTIONSPACESIZE)],
                                               dtype = np.float32)

                    #print("Child M = %.5f " % (childNode.M  @ actionWeighChild))
                    #print("Child J = %.5f " % (childNode.J  @ actionWeighChild))
                    #print("This Reward = %.5f " % (childNode.reachingReward))
                    #print("Child prob = %.5f " % childNode.reachingProb)
                    self.J += (childNode.J @ actionWeighChild * settings.DISCOUNTING) * info[2] * canBeReached

                    self.M += (settings.DISCOUNTING**2 * (childNode.M  @ actionWeighChild) + 
                               2 * settings.DISCOUNTING * (childNode.J  @ actionWeighChild) * info[1]
                              ) * info[2] * canBeReached

                    #print("Update: Reward at this node = %.4f, %.4f" %(self.J[0], self.J[1]))
                    #print("Update: Reward**2 at this node = %.4f, %.4f" %(self.M[0], self.M[1]))
                #print("Node Done: " + str(self))
        return self.GetState()
    
    def ComputePolicyGradientEPG(self, theta, zFun, parentNode, cumulated, policy):
        # cumulated = [r, z, p]
        ouput_J = np.zeros(theta.shape)
        ouput_V = np.zeros(theta.shape)
        
        if parentNode == None:
            for child in self.child:
                J, V = child.ComputePolicyGradientEPG(theta, zFun, self, [0., np.zeros(theta.shape), 1.], policy)
                ouput_J += J
                ouput_V += V
            return ouput_J, ouput_V
        
        for info in self.parent[parentNode]:
            newR = cumulated[0] + info[1]
            newZ = cumulated[1] + zFun(theta, parentNode.state, info[0])
            newP = cumulated[2] * info[2] * policy(parentNode.height, parentNode.state)[info[0]]

            if len(self.child) == 0:
                ouput_J += newR * newZ * newP
                ouput_V += newR**2 * newZ * newP
            for child in self.child:
                newCumulated = [newR, newZ, newP]
                J, V = child.ComputePolicyGradientEPG(theta, zFun, self, newCumulated, policy)
                ouput_J += J
                ouput_V += V
        return ouput_J, ouput_V
    
    def UpdateParent(self, parentNode, info):
        # info: reaching action, reaching reward, reaching Prob
        if parentNode in self.parent.keys():
            self.parent[parentNode].append(info)
            return False
        else:
            self.parent[parentNode] = [info]
            return True
            
    
    def AddVisitProb(self, x):
        self.visitProb += x
        
    def GetVisitProb(self):
        return self.visitProb

    def GetState(self):
        return self.state

    def GetMoment(self, action):
        return self.J[action], self.M[action]
    
    def __str__(self):
        return settings.StateToString(self.GetState())