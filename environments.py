import gym
from gym import spaces
import numpy as np
from scipy.stats import norm
import settings

class SomeEnvironment(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['console']}


    def __init__(self, 
                 horizon = 50, 
                 bEndAtRecurrentState = True):

        super(SomeEnvironment, self).__init__()
        
        # ==== Dynamics/Reward ====
        
        # ==== Action ====
        #n_actions = 2
        #self.action_space = spaces.Discrete(n_actions)

        
        # ==== State ====
        self.state = np.array([])
        self.financialPerformance = 0
        #self.initialState =np.zeros(self.N + 2)    # place holder        
        
        # this can be described by Box space
        #self.observation_space = spaces.Box(low=-1, high=1,
        #                                   shape=(self.N + 2,), dtype=np.float32)

        
        # ==== Ending ====
        # stop: either stop at t = T; or reach the recurrent state
        self.timeStep = 0
        self.horizon = horizon
        self.bEndAtRecurrentState = bEndAtRecurrentState
        
        
    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        # Initialize
        self.timeStep = 0
        #self.financialPerformance = 0
        
        return self.GetState()
    
    
    def step(self, action, debug = False):
        # ========= Action ============
        
        
        # ========== State Update ===============
        

        # ================ prepare for output =================
        output = np.array([])

        # have we reached the horizon
        self.timeStep += 1
        done = True
        
        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = np.log(1)
        
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        
        return output, reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("step: " + str(self.timeStep))
        
        print(self.liquidAsset)
        print(self.illiquidAssets)
        
        print(".")

    def close(self):
        pass
    

    def GetState(self):
        return self.state

    def GetFinancialPerformance(self):
        return self.financialPerformance
    
class SPERLPM(SomeEnvironment):
    """
    Custom Environment that follows gym interface.
    """

    metadata = {'render.modes': ['console']}

    def __init__(self, miu, sigma=0.3, horizon = 100): #increasing = 1):
        super(SPERLPM, self).__init__(horizon = horizon, bEndAtRecurrentState = False)
        
        # ==== Dynamics/Reward ====
        self.riskFreeInterest = 0.02
        self.volatility = sigma #0.3
        self.possibleMean = [-0.2, 0.2]
        self.mean = miu # self.possibleMean[increasing]
        self.gamma = 1.2   #mean-variance criterion parameter
        
        
        # ==== Action ====
        self.action_space = spaces.Box(low=0, high=1,
                                       shape=(1,), dtype=np.float32)

        
        # ==== State ====
        self.wealth = 1
        self.observation_space = spaces.Box(low=-2, high=2,
                                       shape=(1,), dtype=np.float32)

        
        # ==== Ending ====
        # stop: either stop at t = T; or reach the recurrent state
        self.timeStep = 0   # from 0 to 99
        self.horizon = horizon
        

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        
        super(SPERLPM, self).reset()
        self.financialPerformance = 1
        self.wealth = 1
        return self.getState()

    def step(self, action, debug = False):
        # Generate randomness in market
        mu = self.mean/self.horizon
        sigma = (self.volatility**2 / self.horizon)**0.5
        y = np.random.normal(mu, sigma, 1)[0]
        
        
        ### Assume shorting and borrowing are not allowed ####
        # meaning action cannot be greater than wealth
        #action = np.clip(action, 0, 1)
        
        # ========== State Update ===============
        wealthPast = self.wealth
        self.wealth += self.wealth * (self.riskFreeInterest/self.horizon +  action * (y - self.riskFreeInterest/self.horizon))
        # reward signal
        reward = self.wealth - wealthPast #np.log(self.wealth/wealthPast)
        if np.isnan(reward):
            print("===========")
            print(self.wealth)
            print(wealthPast)
            print("===========")
            
        # but what if it becomes negative.
        # it happens when 
        
        
        # ================ Update Financial Performance ========
        self.financialPerformance = self.wealth
        
        # ================ prepare for output =================
        output = self.getState()
        
        # have we reached the horizon
        self.timeStep += 1
        done = self.timeStep == self.horizon - 1
        
        
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        
        return output, reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("step: " + str(self.timeStep))
        
        print(self.wealth)        
        print(".")

    def close(self):
        pass
    
    
    def getState(self):
        return np.array([self.timeStep, self.wealth, 1])
    
    
class PM(SomeEnvironment):
    """
    Custom Environment that follows gym interface.
    portfolio management environment
    """
    # Define constants for clearer code
    HOLD = 0
    INVEST = 1
    
    optionsList = [(0,(0,0)),
                  (0,(1,0)),
                  (0,(0,1)),
                  (0,(1,1)),
                  (1,(0,0)),
                  (1,(1,0)),
                  (1,(0,1)),
                  (1,(1,1))]  # (a, w)
    
    def OptionsProb(self, options):
        bSwithch, bDefault = options[1]
        
        return (self.pRisk if bDefault else (1 - self.pRisk)) * (self.p if bSwithch else (1- self.p))
    
    def __init__(self, 
                 parasForEnv = {"interestRate": [2, 0.5,  1],
                                 "horizon" : 5,
                                 "illiquidPeriod" : 2,
                                 "pDefault" : 0.1,
                                 "pSwitch" : 0.2},
                 bEndAtRecurrentState = False): 
        super(PM, self).__init__(horizon = parasForEnv["horizon"], bEndAtRecurrentState = bEndAtRecurrentState)
        
        # ==== Dynamics/Reward ====
        # liquid asset interest gain
        self.rLiquid = parasForEnv["interestRate"][2]
        
        # non-liquid asset interest gain
        self.N = parasForEnv["illiquidPeriod"]
        self.p = parasForEnv["pSwitch"]
        
        rHigh = parasForEnv["interestRate"][0]
        rLow = parasForEnv["interestRate"][1]
        
        self.rIL_choice = [rHigh, rLow]
        rIL_dis = np.array([0.5, 0.5])
        self.rIL_index = 1 if len(settings.INITIAL_STATES) == 1  else np.random.choice(2, p = rIL_dis)
        
        ltLow = rLow - (rHigh * self.p + rLow * (1 - self.p))
        ltHigh = rHigh - (rHigh * (1 - self.p) + rLow * self.p)
        self.lastTermChoice = (ltHigh, ltLow)
        self.timeInState = 0
        
        # non-liquid asset default rate
        self.pRisk = parasForEnv["pDefault"]
        
        # ==== Action ====
        # Action: fixed fraction of total wealth per investment
        # it is denoted as kappa in the paper.
        self.alpha = 0.2
        # using discrete actions, we have two: invest or hold
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)

        # ==== State ====
        # initialization for state related variable
        self.liquidAsset = 1.0
        self.illiquidAssets = np.zeros(self.N, dtype = np.float32)
        #self.expected_rIL = rIL_dis @ np.array(self.rIL_choice)    # place holder
        
        self.initialState =np.zeros(self.N + 2)    # place holder        
        # this can be described by Box space
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.N + 2,), dtype=np.float32)

        
        # ==== Ending ====
        # stop: either stop at t = T; or reach the recurrent state
        # inheritance     


    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        
        super(PM, self).reset()
        self.liquidAsset = 1.0
        self.illiquidAssets = np.zeros(self.N, dtype = np.float32)
        self.timeInState = 0
        
        self.financialPerformance = 1.0
        rIL_dis = np.array([0.5, 0.5])
        self.rIL_index = 1 if len(settings.INITIAL_STATES) == 1  else np.random.choice(2, p = rIL_dis)
        self.initialState = self.getState()

        return self.getState()

    def NextState(self, state, time,  action, w):
        bSwitch, bDefault = w
        if bSwitch:
            bUp = state[-2] < 0
        else:
            bUp = state[-2] > 0
        # 即便有相等的node，但也没有共用的node（用计算时间换取易读性）
        liquidAsset = state[0]
        illiquidAssets = state[1:-2].copy()
        if liquidAsset < 0:
            print("load")
            print(illiquidAssets)
            print(liquidAsset)
            raise ValueError
        
        if action == 0 or liquidAsset < self.alpha:
            toBeInvest = 0
        elif action == 1:
            toBeInvest = self.alpha
    
        # mature
        liquidAsset += illiquidAssets[0] if not bDefault else 0 
        illiquidAssets = np.roll(illiquidAssets, -1)      
        if liquidAsset < 0:
            print("mature")
            print(illiquidAssets)
            print(liquidAsset)
            raise ValueError
        
        # action of investing
        liquidAsset -= toBeInvest
        illiquidAssets[-1] = toBeInvest
        if liquidAsset < 0:
            print("invest")
            print(temp)
            print(self.alpha)
            print(toBeInvest)
            print(illiquidAssets)
            print(liquidAsset)
            raise ValueError
        
        # increase in value        
        liquidAsset *= 1 + self.rLiquid
        illiquidAssets =illiquidAssets * ( 1 + (self.rIL_choice[0] if bUp else self.rIL_choice[1]))
        if liquidAsset < 0:
            print("interest")
            print(illiquidAssets)
            print(liquidAsset)
            raise ValueError
        
        total = liquidAsset + np.sum(illiquidAssets)
        if total < 0:
            print("total:", total)
            print(illiquidAssets)
            print(liquidAsset)
            raise ValueError
        
        if time == self.horizon - 1:
            reward = np.log(total * (1 - self.pRisk))
        else:
            reward = np.log(total)
        
        liquidAsset /= total
        illiquidAssets = illiquidAssets / total
        
        output = np.concatenate(([liquidAsset],
                                 illiquidAssets, 
                                 [-state[-2] if bSwitch else state[-2],
                                  time + 1])
                               ).astype(np.float32)
        # Null reward everywhere except when reaching the goal (left of the grid)
        
        
        return output, reward, False
    
    def step(self, action, debug = False):
        # ========= Action ============
        # invest in iliquid asset
        if action == self.HOLD or self.liquidAsset < self.alpha:
            toBeInvest = 0
        elif action == self.INVEST:
            toBeInvest = self.alpha
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))
        
        
        # ========== State Update ===============
        # change in time dependent interest rate
        self.rIL_index = (self.rIL_index + (1 if np.random.rand()<self.p else 0)) % 2
        
        # mature
        matureReturn = np.random.choice([0, self.illiquidAssets[0]], p = [self.pRisk, 1 - self.pRisk]) 
        self.liquidAsset += matureReturn
        # self.illiquidAssets[0] = 0 # make no difference
        self.illiquidAssets = np.roll(self.illiquidAssets, -1)        
        
        # action of investing
        self.liquidAsset -= toBeInvest
        self.illiquidAssets[-1] = toBeInvest

        # increase in value        
        self.liquidAsset *= 1 + self.rLiquid
        self.illiquidAssets *= 1 + self.rIL_choice[self.rIL_index]      
        if debug:
            print(self.liquidAsset, end=', ')
            print(self.illiquidAssets)
        
        total = self.liquidAsset + np.sum(self.illiquidAssets)            
        self.liquidAsset /= total
        self.illiquidAssets /= total
        self.timeInState += 1
        # ================ Update Financial Performance ========
        self.financialPerformance *= total
        
        # ================ prepare for output =================
        output = self.getState()

        # have we reached the horizon
        self.timeStep += 1
        done = (self.bEndAtRecurrentState and np.array_equal(output, self.initialState)) or self.timeStep == self.horizon            
        
        # Null reward everywhere except when reaching the goal (left of the grid)
        if self.timeStep == self.horizon - 1:
            reward = np.log(total * (1 - self.pRisk))
        else:
            reward = np.log(total)        
        
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        
        return output, reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("step: " + str(self.timeStep))
        
        print(self.liquidAsset)
        print(self.illiquidAssets)
        
        print(".")

    def close(self):
        pass
    
    def getN(self):
        return self.N
    
    
    def getState(self):
        return np.concatenate(([self.liquidAsset],
                                 self.illiquidAssets, 
                                 [self.lastTermChoice[self.rIL_index],
                                  self.timeInState])
                               ).astype(np.float32)
    
    

class OptimalExecution(SomeEnvironment):
    """
    Custom Environment that follows gym interface.
    """
    totalHolding = 10**6       # denoted as Y in the paper
    initialStockPrice = 50     # denoted as S_0 in the paper
    
    # disretization of state
    NUMBER_OF_BINS_PRICE = 200
    DISCRETISZE_FACTOR_PRICE = 1
    NUMBER_OF_BINS_N = 10
    NUMBER_OF_BINS_X = 10
    
    optionsList = []  # (a, w)
    
    def computeW(self, w):
        ls = np.linspace(-4, 4, self.NUMBER_OF_BINS_W + 1)
        return ls[w] + (ls[w+1] - ls[w]) / 2
    
    def OptionsProb(self, options):
        w = options[1]
        ls = np.linspace(-4, 4, self.NUMBER_OF_BINS_W + 1)
        return norm.cdf(ls[w+1]) - norm.cdf(ls[w])
    
    def __init__(self, 
                 parasForEnv = {'horizon': 5,
                                'sigma' : 0.019,
                                'numW': 4}, 
                 gamma = 25, 
                 eta = 25, 
                 epsilon = 0.0625): 
        super(OptimalExecution, self).__init__(horizon = parasForEnv['horizon'], bEndAtRecurrentState = False)
        
        
        self.NUMBER_OF_BINS_W =  parasForEnv['numW'] #, 4 is for testing
        if 'eta' in parasForEnv.keys():
            self.eta =  parasForEnv['eta'] * 10 ** (-7)
        else:
            self.eta = eta * 10 ** (-7)  # Impact at 1% of market
                             # depend on market micro-structure
            
        
            
        self.epsilon = epsilon   # bid-ask spread = 1/8   for computation of expected value
                                 # fixed cost of selling
        
        if 'gamma' in parasForEnv.keys():
            self.gamma = parasForEnv['gamma'] * 10 ** (-8)   # daily volumn 5 million shares
                                 # linear permanent price impact coefficent
        else:
            self.gamma = gamma * 10 ** (-8)   # daily volumn 5 million shares
                                 # linear permanent price impact coefficent
        
        
        self.lamb = 10 ** (-6)   # static holdings 11000 shares ??? 
                                 # Lagrange multiplier, risk-aversion
        
        # ==== Dynamics/Reward ====
        self.sigma = parasForEnv['sigma']   # 30% annual volatility
        
        # now we assume no information about annual growth
        # self.miu = 0.02   # 10% annual growth        
        

        # ===== Action =====
        self.action_space = spaces.Discrete(settings.ACTIONSPACESIZE)
        
        self.optionsList = [None] * settings.ACTIONSPACESIZE * self.NUMBER_OF_BINS_W  # (a, w)        
        for i in range(settings.ACTIONSPACESIZE):
            for j in range(self.NUMBER_OF_BINS_W):
                self.optionsList[j + i*self.NUMBER_OF_BINS_W] = (i, j)  # (a, w)
        
        # ==== State ====
        self.holding = self.totalHolding   # X, inititial holdings
        self.stockPrice = self.initialStockPrice   # S_0 initial stock price
        self.logReturn = 0
        
        self.observation_space = spaces.Box(low=np.array([-np.inf, 0, 0]), high=np.array([np.inf, 1.0, 1.0]), dtype=np.float32)
        
        # ==== Ending ====
        # stop: either stop at t = T; or 
        #       reach the recurrent state (impossible)

        self.T = parasForEnv['horizon']   # T, time horizon
        self.N = parasForEnv['horizon']   # N, number of time period

        self.remainingNumTrade = self.N   # from 0 to N-1
        self.tau = self.T / self.N   # length of one trade interval
        
        # other info
        self.financialPerformance = 0
        
    def getState(self):
        try:
            return np.array([self.logReturn * self.DISCRETISZE_FACTOR_PRICE * self.NUMBER_OF_BINS_PRICE // 1,
                             self.remainingNumTrade / self.N * self.NUMBER_OF_BINS_N // 1,
                             self.holding / self.totalHolding * self.NUMBER_OF_BINS_X // 1], dtype = np.intc)
        except ValueError:
            print(self.stockPrice)
            raise ValueError

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        
        super(OptimalExecution, self).reset()
        self.financialPerformance = 0
        self.remainingNumTrade = self.N
        self.holding = self.totalHolding   # X, inititial holdings
        self.stockPrice = self.initialStockPrice   # S_0 initial stock price
        self.logReturn = 0
        return self.getState()
    
    def NextState(self, state, time, action, w):
        w = self.computeW(w)
        remainingNumTrade = state[1] / self.NUMBER_OF_BINS_N * self.N
        holding = state[2] / self.NUMBER_OF_BINS_X * self.totalHolding
        stockPrice = np.exp(state[0] / self.NUMBER_OF_BINS_PRICE / self.DISCRETISZE_FACTOR_PRICE) * self.initialStockPrice
        
        
        if remainingNumTrade ==1:
            sharesSold = holding
        else:
            sharesSold = holding * action / (settings.ACTIONSPACESIZE - 1)
            
        # selling
        pricePerShare = stockPrice - self.h(sharesSold / self.tau) # temporary price impact
        holding -= sharesSold
        
        # update price
        stockPrice += self.sigma * self.tau ** 0.5 * w * stockPrice  # market fluctuation
        stockPrice -= self.tau * self.g(sharesSold / self.tau)   # permanent price impact
        
        logReturn = np.log(stockPrice / self.initialStockPrice)
        remainingNumTrade -= 1
        
        # reward signal(negative shortfall, to maximize the reward)
        reward = (pricePerShare - self.initialStockPrice) * sharesSold / self.initialStockPrice / self.totalHolding
        
        done = remainingNumTrade == 0
        output = np.array([logReturn * self.NUMBER_OF_BINS_PRICE * self.DISCRETISZE_FACTOR_PRICE // 1,
                         remainingNumTrade / self.N * self.NUMBER_OF_BINS_N // 1,
                         holding / self.totalHolding * self.NUMBER_OF_BINS_X // 1], dtype = np.intc)
        return output, reward, done
    
    def step(self, action, debug = False):
        # Generate randomness in market
        w_p = np.array([self.OptionsProb((0,i)) for i in range(4)])
        w_p /= sum(w_p)
        w = self.computeW(np.random.choice(np.arange(self.NUMBER_OF_BINS_W), p = w_p)) #np.random.normal()
        
        # ========== State Update ===============
        # action is in range(10), percentage of remaining holdings sold * 10
        if self.remainingNumTrade ==1:
            sharesSold = self.holding
        else:
            sharesSold = self.holding * action / (settings.ACTIONSPACESIZE - 1)
        
        # selling
        pricePerShare = self.stockPrice - self.h(sharesSold / self.tau) # temporary price impact
        self.holding -= sharesSold
        
        # update price
        self.stockPrice += self.sigma * self.tau ** 0.5 * w * self.stockPrice  # market fluctuation
        self.stockPrice -= self.tau * self.g(sharesSold / self.tau)   # permanent price impact
        
        
        self.logReturn = np.log(self.stockPrice / self.initialStockPrice)
        self.remainingNumTrade -= 1
        
        # reward signal(negative shortfall, to maximize the reward)
        reward = (pricePerShare - self.initialStockPrice) * sharesSold / self.initialStockPrice / self.totalHolding
        
        self.financialPerformance += pricePerShare * sharesSold
        
        # ================ prepare for output =================
        output = self.getState()
        
        # have we reached the horizon
        done = self.remainingNumTrade == 0
        
        
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        return output, reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        
        print(".")

    def close(self):
        pass
    
    def g(self, v):
        # permanent price impact
        return self.gamma * v
    
    def h(self, v):
        # temp price impact
        return self.epsilon * np.sign(v) + self.eta * v 