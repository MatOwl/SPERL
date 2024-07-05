import numpy as np

class Agents():
    def __init__(self, Ntheta, actionSpace, PolicyFunction, initTheta = np.array([])):
        # Initialize values and parameter
        
        # Ntheta: size of parameter vector
        # actionSpace: action space
        
        # Policy: function that compute distrubition of action according to state and theta
            # take array with dimension Na, and array with same dimension of state
            # return an array with dimension Na, sum to 1.
        # initTheta: the parameter vector that we start with
        if type(Ntheta) == int:
            self.theta = np.random.rand(Ntheta) if initTheta.size ==0 else np.copy(initTheta)
        else:
            self.theta = np.random.rand(*Ntheta) if initTheta.size ==0 else np.copy(initTheta)
            
        self.Ntheta = Ntheta
        self.actionSpace = actionSpace
        
        self.policy = PolicyFunction
        
        self.totalStep = 1
        
    
    def ValidatePolicy(self, policy, state):
        try:
            result = policy(self.theta, state)
            if np.sum(result) != 1:
                raise ValueError("Output of policy function does not sum to one")
        except:
            raise ValueError("policy function with wrong format")
    
    
    def DecideAction(self, state):
        # decide an action according to policy function
        #[not take action, take action]
        pi = self.policy(self.theta, state)
        if any(np.isnan(pi)):
            print("pi contain Nan. Print(theta, state, pi)")
            print(self.theta)
            print(state)
            print(pi)
            self.policy(self.theta, state, True)
        output = np.random.choice(self.actionSpace, p=pi)
        return output

    
    def StepUpdate(self, state, action, reward):
        pass
    
    def UpdateEps(self, debug = False): 
        pass
    
    
    def ClearStep(self):
        self.totalStep = 1
    

class EPG(Agents):
    # constainted optimization
    b = 20
    flip = False
    stepSequenceConvRate = 0

    # epsilon constraint softMax
    epsilon = 0.05

    
    # to be removed
    N = 0
    
    def __init__(self, thetaShape, actionSpace, 
                 PolicyFunction, 
                 ZUpdate, 
                 objectiveType = 0, 
                 initTheta = np.array([]), 
                 gamma = 1.2,
                 epsilon = 0.05):
        # Initialize values and parameter
        
        # thetaShape: shape of parameter vector
        # actionSpace: actionSpace
        
        # Policy: function that compute distrubition of action according to state and theta
            # take array with dimension Ntheta, and array with same dimension of state
            # return an array with dimension of actionSpace, sum to 1.
        # initTheta: the parameter vector that we start with
        
        # objectiveType: 0 constraint, 1 other
        
        Agents.__init__(self, thetaShape, actionSpace, PolicyFunction, initTheta)
        
        self.thetaBackUp = self.theta.copy()
        # Set hyperParameters
        self.fastStepSize = 0.05
        self.slowStepSize = 0.05
        
        # algo specific parameters
        self.mode = objectiveType
        self.J = 0
        self.V = 1
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.zUpdate = ZUpdate
        
        # initialize temp memory
        self.z = np.zeros(thetaShape)
        self.R = 0
        self.printCounter = 10
            
    def ThetaRollBack(self):
        self.theta = self.thetaBackUp.copy()
        
    def ThetaBackUp(self):
        self.thetaBackUp = self.theta.copy()
    
    def StepUpdate(self, state, action, reward):
        self.z += self.zUpdate(self.theta, state, action)
        self.R += reward
    
    def G(self, x):
        # penalty function
        return (max(0, 2*x ))
    
    def TrueUpdate(self, nabla_J, nabla_V):
        # nabla_J, nabla_V are gradient of J, V (array) with respect to theta
        temp = 0
        # assume mode != 0
        
        temp = self.slowStepSize / self.totalStep  * (nabla_J - self.gamma * nabla_V)
        self.theta += temp
        #self.theta /= np.max(self.theta, axis=None)
                                                          
    def UpdateEps(self, trajectory, debug = False, bTrain = True): 
        # follow and go through the trajectory
        for state, action, reward in trajectory:
            self.StepUpdate(state, action, reward)
        output = self.R
            
        if bTrain:
            # do the final update
            temp = 0
            if self.flip:
                if self.mode == 0:
                    pass
                else:
                    """
                    temp = self.slowStepSize / self.totalStep / (self.V**0.5) * (self.R - 
                                                                                 (self.J*self.R**2 - 
                                                                                  2* self.R * self.J**2)/(2*self.V)) * self.z
                      """
                    temp = self.slowStepSize / self.totalStep  * (self.R - 
                                                                 self.gamma * (self.R**2 - 2*self.J*self.R)) * self.z
                if not np.isnan(temp).any():
                    self.theta += temp
                else:
                    print("Warning: nan update")
            self.V += self.fastStepSize / self.totalStep * (self.R**2 - self.J**2 - self.V)
            self.J += self.fastStepSize / self.totalStep * (self.R - self.J)

            # debugging log
            if self.printCounter and debug:
                print("J = " + str(self.J))
                print("V = " + str(self.V))
                print("R = " + str(self.R))
                print("Something: " + str(temp))
                print("z = " + str(self.z))
                print("theta = " + str(self.theta))
                print(".")
                self.printCounter -= 1

            self.flip  = not self.flip
            self.totalStep += self.stepSequenceConvRate
        
        self.z = np.zeros(self.Ntheta)
        self.R = 0
        return output, self.J, self.V
        
    def SetConstraint(self, b):
        self.b = b
    
    def SetStepSize(self, alpha, beta, delta):
        self.fastStepSize = alpha  # V, J
        self.slowStepSize = beta   # theta
        self.stepSequenceConvRate = delta