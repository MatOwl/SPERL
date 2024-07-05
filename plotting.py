import matplotlib.pyplot as plt
import numpy as np

import settings

colorDict = {
    "PreCom":"y",
    "SPE":"b",
    "SPERL":"r",
    "EPG":"g",
    "EPG_TU":"m",
    "Naive":"k"
}

# df.sort_values(by=['col1', 'col2'])

def plotAction(imgLocation, 
               bigDF, 
               nonZeroCheckAlgoList = ["PreCom", "SPE", "SPERL", "EPG", "EPG_TU"],
               bPlot = False, 
               bTminusOneOnly = False, 
               bImportantOnly = True):
    plt.figure(figsize=(15, 5), constrained_layout=True)
    
    t = (settings.HORIZON - 1) if bTminusOneOnly else 0
    df = bigDF[bigDF["t"] >= t]

    if not nonZeroCheckAlgoList:
        tempList = np.array([True] * df.shape[0])
    else:
        tempList = np.array([False] * df.shape[0])
    for agent in nonZeroCheckAlgoList:
        if ("A_" + agent) not in df.columns:
            continue
        tempList = np.logical_or(tempList, df["P_" + agent]>0.01)
    df = df[tempList]
        
    timeCount = df['t'][tempList].value_counts().sort_index()
    tempCount = 0
    for i in timeCount:
        tempCount += i
        plt.axvline(x=tempCount-0.5, color='r', linestyle='--', alpha = 0.3)
    
    for agent in ["PreCom", "SPE", "SPERL", "EPG", "EPG_TU"]:
        if ("A_" + agent) in df.columns:
            plt.plot(np.arange(df.shape[0]), df["A_" + agent], colorDict[agent]+'o', label = agent, alpha = 0.5)
    plt.legend(loc = "right")
    plt.title("Action for each time-state" + ("at t = T - 1" if bTminusOneOnly else ""))
    
    if bPlot:
        plt.show()
    else:
        plt.savefig(imgLocation + "ActionPlot_" + ("atTminus1" if bTminusOneOnly else ""), dpi=200)
        
    
def plotUtility(imgLocation, 
                bigDF, 
                nonZeroCheckAlgoList = ["PreCom", "SPE", "SPERL", "EPG", "EPG_TU"],
                bPlot = False, 
                bTminusOneOnly = False, 
                bRelativeToBase = False, 
                bProbAdj = False):
    plt.figure(figsize=(15, 5), constrained_layout=True)
    
    t = (settings.HORIZON - 1) if bTminusOneOnly else 0
    df = bigDF[bigDF["t"] >= t]
    
    if not nonZeroCheckAlgoList:
        tempList = np.array([True] * df.shape[0])
    else:
        tempList = np.array([False] * df.shape[0])
    for agent in nonZeroCheckAlgoList:
        if ("A_" + agent) not in df.columns:
            continue
        tempList = np.logical_or(tempList, df["P_" + agent]>0.01)
    df = df[tempList]
        
    timeCount = df['t'][tempList].value_counts().sort_index()
    tempCount = 0
    for i in timeCount:
        tempCount += i
        plt.axvline(x=tempCount-0.5, color='r', linestyle='--', alpha = 0.3)
    
    if bRelativeToBase:
        for agent in ["PreCom", "SPE", "SPERL", "EPG", "EPG_TU"]:
            if ("A_" + agent) in df.columns:
                plt.plot(np.arange(df.shape[0]), df["U_" + agent] - df["U_Naive"], colorDict[agent]+'-', label = agent, alpha = 0.5)
    else:
        for agent in ["PreCom", "SPE", "SPERL", "EPG", "EPG_TU"]:
            if ("A_" + agent) not in df.columns:
                continue
            if bProbAdj:
                tobePlot = df["U_" + agent] * df["P_" + agent]
            else:
                tobePlot = df["U_" + agent]
            #plt.bar(np.arange(tobePlot.shape[0]), tobePlot, color = colorDict[agent], label = agent, alpha = 0.3)
            plt.plot(np.arange(df.shape[0]), tobePlot, colorDict[agent]+'-', label = agent, alpha = 0.5)

    #plt.plot(range(len(SPERLListJ)), PreComOptListU, 'y-', label = "PreCOm", alpha = 1)
    plt.legend(loc = "upper right")
    plt.title("Utility for Each Time-State Pair "+
              ("Compared with baseline " if bRelativeToBase else "") + 
              ("at t = T - 1" if bTminusOneOnly else "") + 
              ("Prob adjusted" if bProbAdj else "")
             )

    if bPlot:
        plt.show()
    else:
        plt.savefig(imgLocation + 
                    "UPlot_" +
                    ("ComparedWithBaseline_" if bRelativeToBase else "")+ 
                    ("atTminus1" if bTminusOneOnly else "") + 
                    ("Prob adjusted" if bProbAdj else ""), dpi=200)
    
def PlotVisitation(imgLocation, 
                   bigDF,
                   nonZeroCheckAlgoList = ["PreCom", "SPE", "SPERL", "EPG", "EPG_TU"],
                   bTminusOneOnly = False, 
                   bPlot = False):
    plt.figure(figsize=(15, 5), constrained_layout=True)
    
    t = (settings.HORIZON - 1) if bTminusOneOnly else 0
    df = bigDF[bigDF["t"] >= t]
    
    t = (settings.HORIZON - 1) if bTminusOneOnly else 0
    df = bigDF[bigDF["t"] >= t]
    if not nonZeroCheckAlgoList:
        tempList = np.array([True] * df.shape[0])
    else:
        tempList = np.array([False] * df.shape[0])
    for agent in nonZeroCheckAlgoList:
        if ("A_" + agent) not in df.columns:
            continue
        tempList = np.logical_or(tempList, df["P_" + agent]>0.01)
    df = df[tempList]

    
    timeCount = df['t'][tempList].value_counts().sort_index()
    tempCount = 0
    for i in timeCount:
        tempCount += i
        plt.axvline(x=tempCount-0.5, color='r', linestyle='--', alpha = 0.3)
    
    for agent in ["PreCom", "SPE", "SPERL", "EPG", "EPG_TU"]:
        if ("A_" + agent) not in df.columns:
                continue
        plt.plot(np.arange(df.shape[0]), df["P_" + agent], colorDict[agent]+'o', label = agent, alpha = 0.5)
    
    plt.legend(loc = "upper right")
    plt.title("Visitation Probability for Each Time-State Pair "+ ("at t = T - 1" if bTminusOneOnly else ""))
    
    if bPlot:
        plt.show()
    else:
        plt.savefig(imgLocation + "Visitation_" + ("atTminus1" if bTminusOneOnly else ""), dpi=200)

        
import matplotlib as mpl
from matplotlib import cm

axisTitleSize = 30
axisLabelSize = 30
axisTickSize = 23
figureTitleSize = 15
legendSize = 15

figWidth = 7
figHeight = 5

# line width
linewidth = 2
markersize = 10
# color
"""
color option:
https://matplotlib.org/stable/gallery/color/named_colors.html
"""
trueCurveColor = 'blue'
reconstructedCurveColor = 'orange'
coloredRegionColor = "lightcyan"

ComponentColor = ["red",
                  "green",
                  "purple",
                  "pink",
                  "black",
                  "olive"
                 ]

#  change the settings above as you like.
plt.rc('lines', linewidth = linewidth, color='r')
plt.rc('axes', titlesize=axisTitleSize)     # fontsize of the axes title
plt.rc('axes', labelsize=axisLabelSize)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=axisTickSize)    # fontsize of the tick labels
plt.rc('ytick', labelsize=axisTickSize)    # fontsize of the tick labels
plt.rc('legend', fontsize=legendSize)    # legend fontsize
plt.rc('figure', titlesize=figureTitleSize)  # fontsize of the figure title

lineColor = 'grey'
higherUColor = 'springgreen' #'royalblue'
lowerUColor = 'salmon' #'peru'
middleColor = 'lightgrey'

sameActionColor = 'tab:blue'
differentActionColor = 'tab:orange'
cmapActionDiff = mpl.colors.LinearSegmentedColormap.from_list("cmapActionDiff", [(0,sameActionColor), (0.5,middleColor),(1,differentActionColor)])
#cmapActionDiff = plt.get_cmap('bwr')
        
def DrawTree(result,
             graphPath,
             algoIndex, 
             algoSubscript, 
             refAlgo = None, 
             baseLine = 0, 
             thinLineThickness = 0., 
             bPrecom = False,
             bSPEAction = True):
    
    probThreshold = 0.001
    
    n_action = (11 if graphPath == "graph/OptEx/" else 2)
    
    root = result[algoIndex].tree.roots[0]
    policy = result[algoIndex].tree.policy
    nodeSet = list(result[algoIndex].tree.stateTimeSpaceDict.values())

    LayerSets = [[] for i in range(settings.HORIZON)]
    for key in result[algoIndex].tree.stateTimeSpaceDict.keys():
        if key[0] >= settings.HORIZON:
            continue
        LayerSets[key[0]].append(result[algoIndex].tree.stateTimeSpaceDict[key])


    fig = plt.figure(figsize=(figWidth, figHeight), constrained_layout=True)
    for i in range(settings.HORIZON - (1 if graphPath == "graph/OptEx/" else 0)):
        tempList =[]
        tempSum = 0
        # find the max/,min utility at this time state.
        
        for j, node in enumerate(LayerSets[i]):        
            P = result[algoIndex].table.iloc[settings.GetTimeStateIndex(i, node.state), 6]
            if P > probThreshold:
                A = result[algoIndex].table.iloc[settings.GetTimeStateIndex(i, node.state), 2] 
                A_P = policy(i, node.state)
                U = result[algoIndex].table.iloc[settings.GetTimeStateIndex(i, node.state), 5]
                y = j/len(LayerSets[i])-(0.5 if i != 0 else 0)
                
                if refAlgo == None:
                    color = clrFn((U - baseLine) / (result[2 if bPrecom else 1].table.loc[:,"U_SPE"].max() - baseLine))
                    actionDiff = 0
                    if i == 0:
                        plt.text(i+0.1, y-0.005, "%.3f" % U)#((U_diff) / (maxU - minU)))
                    
                elif bSPEAction:

                    if result[algoIndex].tree.bStochasticPolicy:
                        averageAction = A #np.mean([i/(n_action-1) * A[i] for i in range(n_action)])
                        actionDiff = averageAction - result[refAlgo[0]].table.loc[settings.GetTimeStateIndex(i, node.state),
                                                                        "A_"+refAlgo[1]]
                        actionDiff = abs(actionDiff)/(n_action-1)
                        
                        color = cmapActionDiff(actionDiff)      
                        
                    else:
                        actionDiff = A - result[refAlgo[0]].table.loc[settings.GetTimeStateIndex(i, node.state),
                                                                        "A_"+refAlgo[1]]
                        actionDiff = abs(actionDiff)/(n_action-1)
                        color = cmapActionDiff(actionDiff)
                    #plt.text(i+0.1, y-0.005, "%.1f" % actionDiff)#((U_diff) / (maxU - minU)))
                    tempSum += actionDiff * P
                
                else:
                    minU = min([result[algoi].table.iloc[settings.GetTimeStateIndex(i, node.state), 5] for algoi in [1,2,3,4]])
                    maxU = max([result[algoi].table.iloc[settings.GetTimeStateIndex(i, node.state), 5] for algoi in [1,2,3,4]])
                    U_diff = U - result[refAlgo[0]].table.loc[settings.GetTimeStateIndex(i, node.state),"U_"+refAlgo[1]]
                    
                    if U_diff > 0.0001:
                        #color = "blue"
                        actionDiff = None
                        pass
                        #plt.text(i+0.1, y-0.005, "%.3f" % U)#((U_diff) / (maxU - minU)))
                        
                    elif maxU - minU < 0.0001:
                        actionDiff = 0
                        color = cmapActionDiff(actionDiff)
                        #plt.text(i+0.1, y-0.005, "%.3f" % U )#(0))
                    else:
                        #plt.text(i+0.1, y-0.005, "%.3f" % U)#((U_diff) / (maxU - minU)))
                        actionDiff = abs(U_diff) / (maxU - minU)
                        color = cmapActionDiff(actionDiff)              
                
                childPloted = []
                colorCounter = 0
                if i == settings.HORIZON:
                    continue
                    
                if i != settings.HORIZON -1  - (1 if graphPath == "graph/OptEx/" else 0):
                    for child in node.child:
                        ChildA = policy(i+1, child.state)
                        if bSPEAction:
                            if result[algoIndex].tree.bStochasticPolicy:
                                averageAction = result[algoIndex].table.iloc[settings.GetTimeStateIndex(i+1, child.state),
                                                                                2]
                                childActionDiff = averageAction - result[refAlgo[0]].table.loc[settings.GetTimeStateIndex(i+1, 
                                                                                                                          child.state),
                                                                                "A_"+refAlgo[1]]
                                childActionDiff = abs(childActionDiff)/(n_action - 1)
                            else:
                                childActionDiff = ChildA - result[refAlgo[0]].table.loc[settings.GetTimeStateIndex(i+1, child.state),
                                                                                "A_"+refAlgo[1]]
                                childActionDiff = abs(childActionDiff)/(n_action - 1)
                                
                        if result[algoIndex].table.iloc[settings.GetTimeStateIndex(i+1, child.state), 6] < probThreshold:
                            continue
                            
                        index = LayerSets[i+1].index(child)
                        alphaCorrecter = 0.05
                        
                        if result[algoIndex].tree.bStochasticPolicy:
                            for info in child.parent[node]:
                                plt.plot([i+ (0.07 if actionDiff > 0.5 else 0 ), i+1+ (0.07 if childActionDiff > 0.5 else 0 )], 
                                         [y, index/len(LayerSets[i+1])-0.5], 
                                         '-', 
                                         color = lineColor,
                                         alpha = ((alphaCorrecter + A_P[info[0]]*P)/(1+alphaCorrecter)))
                        else:
                            for info in child.parent[node]:
                                plt.plot([i+ (0.07 if actionDiff > 0.5 else 0 ), i+1+ (0.07 if childActionDiff > 0.5 else 0 )], 
                                         [y, index/len(LayerSets[i+1])-0.5], 
                                         '-',
                                         color = lineColor,
                                         alpha = ((alphaCorrecter + 1*P)/(1+alphaCorrecter))
                                         #alpha = 0.25 * (1 if (A == info[0] and P > probThreshold) else thinLineThickness),
                                        )

                plt.plot(i + (0.07 if actionDiff > 0.5 else 0 ), y, marker = 'o' if actionDiff <= 0.5 else '^',
                         c = color, #
                         markersize = markersize,
                         alpha = 1)  # U - UList[i]    
    

    #return nodeLists, edgeDict
    plt.yticks([])
    if graphPath == "graph/OptEx/":
        plt.xlim((0-0.2, (3)+0.2))
        plt.ylim((-0.55, 0.55))
        plt.xticks([0,1,2,3])
    else:
        plt.xlim((0-0.2, (4)+0.2))
        plt.xticks([0,1,2,3,4])
        plt.ylim((-0.6, 0.6)) # PM 1
        
    
    plt.margins(0.12)
    plt.xlabel('t')