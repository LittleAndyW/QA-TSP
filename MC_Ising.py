import random
import matplotlib.pyplot as plt
import numpy as np
import copy 
import math
import time

'''这个就是MCMC模拟，用来模拟某个温度下XY Ising模型分布，
几点注意：
注意1，二维伊辛模型，我们用矩阵来模拟。
注意2，旋转的方向，我们用0到2pi表示吧
算法过程：
一，用一个对称的分布，高斯分布初始化矩阵
二，下面是循环
    （1）产生一个候选的自旋方向，为了连续，假设新产生的自旋方向变化最多比原来变化pi/2
    也就是旋转90度
    （2）计算两个概率，这里热统中的配分函数正比于概率，因此我们用配分函数Z的比值
    Z(变化后)/Z(变化前)=exp(-(E后-E前)/kT) ,这里k是玻尔兹曼常数，T是开尔文温度。令
    这个值是alpha
    (3)判断是都接受*********这个规则有进一步讨论*************
         （3_1)产生一个随机数u在【0,1】之间
         （3_2）如果u<=alhpa,接受候选的，改变此时的自旋状态
         (3_3)如果u>alpha，不接受候选的，不改变此时的自旋状态
inputdata: S :matrix 随机分布,假设已经产生
param: T 温度
       delta 最大的变化度数，默认是90度，也可以调整为其他
outputdata:S
''' 

def MetropolisHastings(S,T,numsOfItera):
    deltamax=0.5  
    sumd=[]
    k = 1  #玻尔兹曼常数 
    for sdw in range(numsOfItera):  
       # k2 = np.random.randint(0,S.shape[0]**2)
        i = np.random.randint(0,S.shape[0])
        j = np.random.randint(0,S.shape[0])
       # print('产生的随机位置是：',i,j)
        #time.sleep(0.1)
        for m in range(1):
            delta = (2*np.random.random()-1)*deltamax
            newAngle = S[i][j]+delta
            sumd.append(abs(delta))
           # print(delta)
            energyBefore = getEnergy(i=i,j=j,S=S,angle=None)
            energyLater = getEnergy(i,j,S=S,angle=newAngle)
            alpha = math.exp(-(energyLater-energyBefore)/(k*T))
            #print(alpha)
           # if alpha>=1:
              #  print('大于1的哦')
            if alpha >=1:
                S[i][j]=newAngle
            elif np.random.uniform(0.0,1.0)<=1.0*alpha:
                S[i][j]=newAngle
    return S
#计算i,j位置的能量 = 与周围四个的相互能之和   
def getEnergy(i,j,S,angle=None):
    width = S.shape[0]
    height = S.shape[1]
   # print('矩阵的宽和高是',width,height)
    top_i = i-1 if i>0 else width-1
    bottom_i = i+1 if i<(width-1) else 0
    left_j = j-1 if j>0 else height-1
    right_j = j+1 if j<(height-1) else 0
    enviroment = [[top_i,j],[bottom_i,j],[i,left_j],[i,right_j]]
  #  print(i,j,enviroment)
    #print(enviroment)
    energy = 0
    if angle ==None:
       # print('angle==None')        
        for num_i in range(0,4,1):            
            energy += -np.cos(S[i][j]-S[enviroment[num_i][0]][enviroment[num_i][1]])
    else:
       # print('Angle')
        for num_i in range(0,4,1):
            energy += -np.cos(angle-S[enviroment[num_i][0]][enviroment[num_i][1]])
    return energy

#S=2*np.pi*np.random.rand(30,30)
#计算整个格子的能力，为了求平均内能
def calculateAllEnergy(S):
    energy =0
    for i in range(len(S)):
        for j in range(len(S[0])):
            energy +=getEnergy(i,j,S)
    averageEnergy = energy/(len(S[0])*len(S))
    return averageEnergy/2
#print(S)
#for j in range(1000):
 #   print(j)
   # MetropolisHastings(S,10)
#这个是输入样本的多少，格子的尺寸，温度。中间那个循环，是随机取迭代的过程
def plot(S):    
    X, Y = np.meshgrid(np.arange(0,S.shape[0]),np.arange(0,S.shape[0]))
    U = np.cos(S)
    V = np.sin(S)
    plt.figure()
    #plt.title('Arrows scale with plot width, not view')
    Q = plt.quiver(X, Y, U, V, units='inches')
#qk = plt.quiverkey(Q, 0.3, 0.3, 1, r'$2 \frac{m}{s}$', labelpos='E',
   #         coordinates='figure')
    plt.show()

def getWeightValue(numsOfSample,sizeOfSample,temperature):

    sumEnergy=[]
    for i in range(numsOfSample):  #产生个数
        print('+++++++正在计算第%s个样本++++++++++'%i)
        S=2*np.pi*np.random.random((sizeOfSample,sizeOfSample))
        #plot(S)
        initialEnergy = calculateAllEnergy(S)
        print('系统的初始能量是:',initialEnergy)
        newS = np.array(copy.deepcopy(S))
        for nseeps in range(3500):            
            newS = MetropolisHastings(newS,temperature,sizeOfSample**2)
            sumEnergy.append(calculateAllEnergy(newS))
        aveEnergy = calculateAllEnergy(newS)
        #plot(newS)
        print('系统的平均能量是:',aveEnergy)
        reshaped = np.reshape(newS,(1,sizeOfSample**2))
        if i ==0:
            s = copy.deepcopy(reshaped)
            continue
        else:
            s = np.row_stack((s,reshaped))
    plt.plot(sumEnergy)
    plt.show()
    return s
#运行getweightValue函数，中间已经把结果会成图了
res = getWeightValue(1,40,2)
#print(len(res))
#画成箭头图表示出现

