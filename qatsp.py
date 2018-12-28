#coding:utf-8
import time
import math
import numpy as np
import os
import random
import matplotlib.pyplot as plt


## 整个自旋的协调 全体のspinの配位
## 自旋的每一维度-是[TROTTER_DIM, TOTAL_TIME, NCITY]由下式表示
def getSpinConfig():

    ## 是一个特洛塔尺寸，旋转的协调在某一时间  あるトロッタ次元の、ある時刻におけるspinの配位
    def spin_config_at_a_time_in_a_TROTTER_DIM(tag):
        config = list(-np.ones(NCITY, dtype = np.int))#生成一维数组，大小为ncity，数据为-1，类型为int
        config[tag] = 1#改变tag位置上的数据为1
        
        return config#返回一个大小为ncity，tag位数为1，其余为-1的一维数组
   

    ## 存在特洛塔尺寸自旋的协调  あるトロッタ次元におけるspinの配位
    def spin_config_in_a_TROTTER_DIM(tag):#tag是一维数组，大小为从1到ncity
        spin = []
        spin.append(config_at_init_time)
        for i in range(TOTAL_TIME-1):
            spin.append(list(spin_config_at_a_time_in_a_TROTTER_DIM(tag[i])))#追加一个大小为ncity，tag位数为1，其余为-1的一维数组
        return spin#返回一个

    spin = []
    for i in range(TROTTER_DIM):  #TROTTER_DIM10
        tag = np.arange(1,NCITY)#生成1-ncity的一维数组
        np.random.shuffle(tag)#将上数组顺序打乱
        spin.append(spin_config_in_a_TROTTER_DIM(tag))
    return spin
##最后生成一个ncity*nicity大小的二维数组

# 选择Trotter的尺寸是最短距离，输出该时刻的路线
def getBestRoute(config):
    length = []
    for i in range(TROTTER_DIM):
        route = []
        for j in range(TOTAL_TIME):
            route.append(config[i][j].index(1))
        length.append(getTotaldistance(route))

    min_Tro_dim = np.argmin(length)   #返回数组最小值的位置
    Best_Route = []
    for i in range(TOTAL_TIME):
        Best_Route.append(config[min_Tro_dim][i].index(1))
    return Best_Route


##统计距离
def getTotaldistance(route):
    Total_distance = 0
    for i in range(TOTAL_TIME):

        #Total_distance += distance(POINT[route[i]],POINT[route[(i+1)%TOTAL_TIME]])/max_distance
        Total_distance += distance(route[i],route[(i+1)%TOTAL_TIME])/max_distance
    return Total_distance


## 真实距离
def getRealTotaldistance(route):
    Total_distance = 0
    for i in range(TOTAL_TIME):
        #Total_distance += distance(POINT[route[i]], POINT[route[(i+1)%TOTAL_TIME]])
        Total_distance += distance(route[i],route[(i+1)%TOTAL_TIME])
    return Total_distance


## 量子蒙特卡罗步骤
def QMC_move(config, ann_para):
    # 两个不同的时间a,b选
    c = np.random.randint(0,TROTTER_DIM) #TROTTER_DIM=10
    a_ = list(range(1,TOTAL_TIME))####添加了list才能用remove
  
    a = np.random.choice(a_)
    a_.remove(a)#排除已经选择了的a
    b = np.random.choice(a_)

    # 在一些＃特洛塔数字c，的时间的，城市P中B，q是  あるトロッタ数cで、時刻a,bにいる都市p,q
    p = config[c][a].index(1)
    
    q = config[c][b].index(1)

    # 初始化的能量差的值  エネルギー差の値を初期化
    delta_cost = delta_costc = delta_costq_1 = delta_costq_2 = delta_costq_3 = delta_costq_4 = 0

    # （7）来回翻转自旋的能量差的第一项  (7)式の第一項についてspinをフリップする前後のエネルギーの差
    for j in range(NCITY):
        l_p_j = distance(p,j)/max_distance
        l_q_j = distance(q,j)/max_distance
        delta_costc += 2*(-l_p_j*config[c][a][p] - l_q_j*config[c][a][q])*(config[c][a-1][j]+config[c][(a+1)%TOTAL_TIME][j])+2*(-l_p_j*config[c][b][p] - l_q_j*config[c][b][q])*(config[c][b-1][j]+config[c][(b+1)%TOTAL_TIME][j])
	#delta_costc=2*sum((-lpj + lqj)*(config[c][][j] - config[c][(a+1)%TOTAL_TIME]))

    # 之前和的能量差之后翻转自旋为等式（7）的第二项  (7)式の第二項についてspinをフリップする前後のエネルギー差
    para = (1/BETA)*math.log(math.cosh(BETA*ann_para/TROTTER_DIM)/math.sinh(BETA*ann_para/TROTTER_DIM))
    delta_costq_1 = config[c][a][p]*(config[(c-1)%TROTTER_DIM][a][p]+config[(c+1)%TROTTER_DIM][a][p]) #-2
    delta_costq_2 = config[c][a][q]*(config[(c-1)%TROTTER_DIM][a][q]+config[(c+1)%TROTTER_DIM][a][q]) #2
    delta_costq_3 = config[c][b][p]*(config[(c-1)%TROTTER_DIM][b][p]+config[(c+1)%TROTTER_DIM][b][p]) #2
    delta_costq_4 = config[c][b][q]*(config[(c-1)%TROTTER_DIM][b][q]+config[(c+1)%TROTTER_DIM][b][q]) #-2

    # 之前和能量差之后＃（7）翻转约类型自旋  (7)式についてspinをフリップする前後のエネルギー差
    delta_cost = delta_costc/TROTTER_DIM+para*(delta_costq_1+delta_costq_2+delta_costq_3+delta_costq_4)

    # 概率min(1,exp(-BETA*delta_cost))在翻转
    if delta_cost <= 0:
        config[c][a][p]*=-1
        config[c][a][q]*=-1
        config[c][b][p]*=-1
        config[c][b][q]*=-1
    elif np.random.random() < np.exp(-BETA*delta_cost):
        config[c][a][p]*=-1
        config[c][a][q]*=-1
        config[c][b][p]*=-1
        config[c][b][q]*=-1

    return config




# 参数的输入
#TROTTER_DIM = int(input("Trotter dimension: "))
#ANN_PARA =  float(input("initial annealing parameter: "))
#ANN_STEP = int(input("Annealing Step: "))
#MC_STEP = int(input("MC step: "))
#BETA = float(input("inverse Temperature: "))

TROTTER_DIM = 10
ANN_PARA =  1.0
ANN_STEP = 400
MC_STEP = 320
BETA = 37
REDUC_PARA = 0.99

"""
获取城市（城市号和x，y坐标）的数据
"""
#FILE_NAME = 'FILE_NAME '

#f = open(os.path.dirname(os.path.abspath(FILE_NAME))+FILE_NAME).read().split("\n")

FILE_NAME = 'FILE_NAME '

f = open('./dist8.txt').read().split("\n")

POINT = []
for i in f:
    POINT.append(i.split(" "))


# 城市数据
NCITY = len(POINT)-1
TOTAL_TIME = NCITY
#for i in range(NCITY):
#   POINT[i].remove(POINT[i][0])##循环remove每个point的第一个数据(X),保留第二个数据(Y)
for i in range(NCITY-1):
    for j in range(2):
        POINT[i][j] = float(POINT[i][j])


def dist_matrix(P,N):
	dist_matrix =[]
	row=[]
	for i in range(N):
		for j in range(N):			
			dist=-math.sqrt((float(P[i][0])-float(P[j][0]))**2 + (float(P[i][1])-float(P[j][1]))**2)
			dist=round(dist,2)
			row.append(dist)
		dist_matrix.append(row)
		row=[]
	return dist_matrix
       

def draw_Point(P,N):
    x=[]
    y=[]
    for i in range(N):
        x.append(P[i][0])
        y.append(P[i][1])
    plt.scatter(x, y, color='b')
    plt.grid(True)
    plt.show()

## 2城市之间距离
#def distance(point1, point2):
    #return math.sqrt((float(point1[1])-float(point2[1]))**2 + (float(point1[0])-float(point2[0]))**2)
## 2城市之间距离
def distance(i,j):
    return d_matrix[i][j]

"""
量子退火模拟
"""
if __name__ == '__main__':
    # 2城市间的距离的最大值
    d_matrix=dist_matrix(POINT,NCITY)
    #draw_Point(POINT,NCITY)
    max_distance =10000
    #求最小值
    for i in range(NCITY-1):
        for j in range(NCITY-1):
            if (max_distance <= distance(i, j))&(distance(i, j)!=0):
                max_distance = distance(i, j)
    max_distance=-600
    # 在初始时间自旋＃协调（肯定城市0是初始时间）  初期時刻におけるspinの配位(初期時刻では必ず都市0にいる)
    config_at_init_time = list(-np.ones(NCITY,dtype=np.int))
    
    config_at_init_time[0] = 1

    print("start...")
    t0 = time.clock()

    np.random.seed(11)
    spin = getSpinConfig()
    

   ################################################################## print(spin)
    LengthList = []
    for t in range(ANN_STEP):
        for i in range(MC_STEP):
            con = QMC_move(spin, ANN_PARA)
            rou = getBestRoute(con)
            length = getRealTotaldistance(rou)
        LengthList.append(length)
        #print("Step:{}, Annealing Parameter:{}, length:{}".format(t+1,ANN_PARA, length))
        ANN_PARA *= REDUC_PARA

#TROTTER_DIM = 10
#ANN_PARA =  1.0
#ANN_STEP = 300
#MC_STEP = 13320
#BETA = 37
#REDUC_PARA = 0.99


    Route = getBestRoute(spin)
    Total_Length = getRealTotaldistance(Route)
    elapsed_time = time.clock()-t0

    print("最短的路线是:{}".format(Route))
    print("最短距离{}".format(Total_Length))
    print("处理时间{}s".format(elapsed_time))

    plt.plot(LengthList)
    plt.show()
