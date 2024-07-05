from main import*

actionOption = {'bComputePreComm' : False,
                'bComputeSPE' : True,
                'bComputeSPERL' : False,
                'bComputeTamar' : False,
                'bComputeTamarTU': True
                }

settings.FORCE_UPDATE = False
SPE_U_List = []
GTrue_U_List = []

for eta in [0, 5, 10, 15,20,25,30,35]:   # 30, 35
    for gamma in [15,20,25,30,35]:  #tea
        for sigma in [0.015, 0.02, 0.025,0.03]: # 
            result = OptExMain("OptEx",
                           selectAlgo = actionOption, 
                           bPrint = False,
                           bPlot = True,
                           bNotSave = False,
                           bContTraining = False,
                           nonZeroCheckAlgoList = [],
                           parasForAlgo = {'bTrueGradient' : True,   # 决定 Training 是 True Gradient
                                           'sFuncForm': "tabular"},   # linear, quadratic, tabular, ..., 
                           parasForEnv = {'horizon': 5,
                                         'sigma' :sigma,
                                         'numW':4,
                                         'gamma' : gamma, # 25 == 2.5 * 10 ** (-7)
                                         'eta': eta
                                         },
                           b_TrainForNewSeed = False,
                           randomSeed = 1000,
                               PolicyFolder = 'seed%d/'%1000
                          )
            SPE_U_List.append((eta, gamma, sigma, result[1].table.iloc[0, 5]))
            GTrue_U_List.append((eta, gamma, sigma, result[2].table.iloc[0, 5]))
            
result = np.array(SPE_U_List)
result[:,-1] -= np.array(GTrue_U_List)[:,-1]

import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
cmapActionDiff = mpl.colors.LinearSegmentedColormap.from_list("cmapActionDiff", [(0,'tab:blue'), (0.5,'lightgrey'),(1,'tab:orange')])


sortDF = pd.DataFrame([result[:,3], np.array(SPE_U_List)[:, -1], np.array(GTrue_U_List)[:, -1], np.array(GTrue_U_List)[:, :-1]]).T
thr = -0
higher  = result[result[:,3]>thr, :]
lower  = result[result[:,3] <thr, :]

plt.tight_layout()
fig = plt.figure(figsize = (10,10), dpi = 500)
ax = fig.add_subplot(projection='3d')
p = ax.scatter(higher[:, 1],higher[:, 2],higher[:, 0], c = higher[:,3], 
               cmap = cmapActionDiff, 
               vmin = -0.015, #np.max(result[:,3]), 
               vmax = 0.015, #np.max(result[:,3]),
               alpha = 1,
               s = 50,
               marker='^'
              )

# fig.colorbar(p)

ax.set_xticks([15,20,25,30,35])
ax.set_yticks([0.015, 0.02, 0.025,0.03], [1.5, 2, 2.5,3])
ax.set_zticks([0,5, 10, 15,20,25,30,35])
ax.invert_xaxis()


ax.scatter([ 25,25 ],
           [ 0.029, 0.015],
           [ 25,25 ],
           alpha = 1,
           s = 100,
           color = 'black',
           marker='x'
              )

fs = 18
ax.set_xlabel('$\zeta$ ($10^{-8}$)', fontsize=fs, labelpad=15)
ax.set_ylabel('$\sigma$ ($10^{-2}$)', fontsize=fs, labelpad=15)
ax.set_zlabel('$\eta$ ($10^{-7}$)', fontsize=fs, labelpad=15)


ax.view_init(elev=10, azim=80, roll=0)
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 0.8, 1, 1]))
ax.tick_params('y', direction='out',pad = 0)
ax.tick_params('x', rotation = 20)
ax.tick_params(labelsize = 15)
plt.subplots_adjust(left = -1)
# fig.colorbar(p)

plt.savefig("paraSweep_OE.png")
plt.show()


actionOption = {'bComputePreComm' : False,
                'bComputeSPE' : True,
                'bComputeSPERL' : False,
                'bComputeTamar' : False,
                'bComputeTamarTU': True
                }

settings.FORCE_UPDATE = False
SPE_U_List2 = []
GTrue_U_List2 = []

for P_sw in [0.1, 0.3 ,0.5, 0.7,0.9]:   #
    for P_def in [0.05, 0.25, 0.45, 0.65, 0.85]:  #
        for R_int in [0.75, 0.8, 0.85, 0.9, 0.95, 1., 1.05, 1.1]: # 
            result = OptExMain("Tamar",
                               selectAlgo = actionOption, 
                               bPrint = False,
                               bPlot = False,
                               bNotSave = False,
                               nonZeroCheckAlgoList = ["SPE","Tamar_TU"],
                               parasForAlgo = {'bTrueGradient' : True,
                                               'sFuncForm': "tabular"},   # linear, quadratic, tabular, ..., 
                               parasForEnv = {
                                              #"interestRate": [1.2, R_int,  1],
                                              "interestRate": [2, R_int,  1],
                                              "horizon" : 5,
                                              "illiquidPeriod" : 3,
                                              "pDefault" : P_def,
                                              "pSwitch" : P_sw},
                               b_TrainForNewSeed = False,
                               randomSeed = 0,
                               PolicyFolder = 'seed0/' # 111
                  )
            SPE_U_List2.append((P_sw, P_def, R_int, result[1].table.iloc[0, 5]))
            GTrue_U_List2.append((P_sw, P_def, R_int, result[2].table.iloc[0, 5]))

result2 = np.array(SPE_U_List2)
result2[:,-1] -= np.array(GTrue_U_List2)[:,-1]

sortDF2 = pd.DataFrame([result2[:,3], np.array(SPE_U_List2)[:, -1], np.array(GTrue_U_List2)[:, -1], np.array(GTrue_U_List2)[:, :-1]]).T

from matplotlib import rc
rc('text', usetex = True)


thr = 0.00
higher2  = result2[result2[:,3]>thr, :]
lower2  = result2[result2[:,3] <thr, :]

plt.tight_layout()
fig = plt.figure(figsize = (10,10), dpi = 500)
ax = fig.add_subplot(projection='3d')

p = ax.scatter(higher2[:, 0],higher2[:, 1],higher2[:, 2], c = higher2[:,3], 
               cmap = cmapActionDiff, 
               vmin = - 0.003, 
               vmax = 0.003,
               alpha = 1,
               s = 50,
               marker='^'
              )

p = ax.scatter(lower2[:, 0],lower2[:, 1],lower2[:, 2], c = lower2[:,3], 
               cmap = cmapActionDiff, 
               vmin = - 0.003, 
               vmax = 0.003,
               alpha = 1,
               s = 50,
               marker='o'
              )

# fig.colorbar(p)

ax.set_xticks([0.1, 0.3 ,0.5, 0.7,0.9])
ax.set_yticks([0.05, 0.25, 0.45, 0.65, 0.85])
ax.set_zticks([0.75, 0.8, 0.85, 0.9, 0.95, 1., 1.05, 1.1])

ax.scatter(
           [ 0.3,0.7],
           [ 0.2, 0.4],
           [ 0.75,1.1 ],
           alpha = 1,
           s = 100,
           color = 'black',
           marker='x'
              )

fs = 18
ax.set_xlabel(r'$P_{switch}$', fontsize=fs, labelpad=20)
ax.set_ylabel(r'$P_{risk}$', fontsize=fs, labelpad=10)
ax.set_zlabel(r'$\underline{r}^{nl}$', fontsize=fs, labelpad=20)


ax.view_init(elev=10, azim=80, roll=0)
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 0.8, 1, 1]))
ax.tick_params('z', direction='out',pad = 10)
# ax.tick_params('y', rotation = 20)
ax.tick_params(labelsize = 15)
plt.subplots_adjust(left = -1)
plt.savefig("paraSweep_PM.png")
plt.show()