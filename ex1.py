#coding:utf-8
import time
import math
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import basefunc as B
import Drawfunc as D
import qafunc as QA



def getSpinConfig():

    ## 是一个特洛塔尺寸，旋转的协调在某一时间  あるトロッタ次元の、ある時刻におけるspinの配位
    def spin_config_at_a_time_in_a_TROTTER_DIM(tag):
        config = list(-np.ones(NCITY, dtype = np.int))#生成一维数组，大小为ncity，数据为-1，类型为int
        config[tag] = 1#改变tag位置上的数据为1
        return config#返回一个大小为ncity，tag位数为1，其余为-1的一维数组

    def spin_config_in_a_TROTTER_DIM(tag):#tag是一维数组，大小为从1到ncity
        spin = []
        spin.append(config_at_init_time)
        for i in range(TOTAL_TIME-1):
            spin.append(list(spin_config_at_a_time_in_a_TROTTER_DIM(tag[i])))
        return spin#返回一个

    spin = []
    for i in range(TROTTER_DIM):
        tag = np.arange(1,NCITY)#生成1-ncity的一维数组
        np.random.shuffle(tag)#将上数组顺序打乱
        spin.append(spin_config_in_a_TROTTER_DIM(tag))
    return spin

def getBestRoute(config):
    length = []
    for i in range(TROTTER_DIM):
        route = []
        for j in range(TOTAL_TIME):
            route.append(config[i][j].index(1))
        length.append(getTotaldistance(route))

    min_Tro_dim = np.argmin(length)
    Best_Route = []
    for i in range(TOTAL_TIME):
        Best_Route.append(config[min_Tro_dim][i].index(1))
    return Best_Route


def getTotaldistance(route):
    Total_distance = 0
    for i in range(TOTAL_TIME):
        Total_distance += distance(route[i],route[(i+1)%TOTAL_TIME])/max_distance
    return Total_distance


def getRealTotaldistance(route):
    Total_distance = 0
    for i in range(TOTAL_TIME):
        Total_distance += distance(route[i], route[(i+1)%TOTAL_TIME])
    return Total_distance


## 量子蒙特卡罗步骤
def QMC_move(config, ann_para):
    # 两个不同的时间a,b选
    c = np.random.randint(0,TROTTER_DIM)
    a_ = list(range(1,TOTAL_TIME))
    a = np.random.choice(a_)
    a_.remove(a)#排除已经选择了的a
    b = np.random.choice(a_)

    # 在一些＃特洛塔数字c，的时间的，城市P中B，q是  あるトロッタ数cで、時刻a,bにいる都市p,q
    p = config[c][a].index(1)
    q = config[c][b].index(1)

    # 初始化的能量差的值  エネルギー差の値を初期化
    delta_cost = delta_costc = delta_costq_1 = delta_costq_2 = delta_costq_3 = delta_costq_4 = 0

    for j in range(NCITY):
        l_p_j = distance(p, j)/max_distance
        l_q_j = distance(q, j)/max_distance
        delta_costc += 2*(-l_p_j*config[c][a][p] - l_q_j*config[c][a][q])*(config[c][a-1][j]+config[c][(a+1)%TOTAL_TIME][j])+2*(-l_p_j*config[c][b][p] - l_q_j*config[c][b][q])*(config[c][b-1][j]+config[c][(b+1)%TOTAL_TIME][j])

    # 之前和的能量差之后翻转自旋为等式（7）的第二项  (7)式の第二項についてspinをフリップする前後のエネルギー差
    para = (1/BETA)*math.log(math.cosh(BETA*ann_para/TROTTER_DIM)/math.sinh(BETA*ann_para/TROTTER_DIM))
    delta_costq_1 = config[c][a][p]*(config[(c-1)%TROTTER_DIM][a][p]+config[(c+1)%TROTTER_DIM][a][p])
    delta_costq_2 = config[c][a][q]*(config[(c-1)%TROTTER_DIM][a][q]+config[(c+1)%TROTTER_DIM][a][q])
    delta_costq_3 = config[c][b][p]*(config[(c-1)%TROTTER_DIM][b][p]+config[(c+1)%TROTTER_DIM][b][p])
    delta_costq_4 = config[c][b][q]*(config[(c-1)%TROTTER_DIM][b][q]+config[(c+1)%TROTTER_DIM][b][q])

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



# 参数的输入''

TROTTER_DIM = 10
ANN_PARA =  1.0
ANN_STEP = 500
MC_STEP = 3320
BETA = 37
REDUC_PARA = 0.99

FILE_NAME = 'FILE_NAME '
#读取点坐标
f = open('./ex1.txt').read().split("\n")
POINT = []

for i in f:
    POINT.append(i.split(" "))
POINT.pop()

#读取点坐标至之间的关系
f = open('./link.txt').read().split("\n")
LINK = []
for i in f:
    LINK.append(i.split(" "))
LINK.pop()

# 城市数据
NCITY = len(POINT)
TOTAL_TIME = NCITY


## 2城市之间距离
def distance(p1, p2):
    return dist_matrix[p1][p2]


"""
量子退火模拟
"""
if __name__ == '__main__':

	matrix=B.creat_dist_matrix(POINT,LINK)
	
	A_Path=B.Floyd(matrix)
	dist_matrix=A_Path[0]  #距离矩阵
	Path=A_Path[1]   #路径矩阵
	
	#getPath(3,5)
	max_distance = 0

	for i in range(NCITY):
		for j in range(NCITY):
			if i == j:
				dist_matrix[i][j]=0

	max_distance = 0
	for i in range(NCITY):
		for j in range(NCITY):
			if max_distance <= distance(i,j):
				max_distance = distance(i,j)


	config_at_init_time = list(-np.ones(NCITY,dtype=np.int))
	config_at_init_time[0] = 1

	print("start...")
	t0 = time.clock()

	np.random.seed(11)
	spin = getSpinConfig()
	LengthList = []
	for t in range(ANN_STEP):
		for i in range(MC_STEP):
			con = QMC_move(spin, ANN_PARA)
			rou = getBestRoute(con)
			length = getRealTotaldistance(rou)
		LengthList.append(length)

		ANN_PARA *= REDUC_PARA

	Route = getBestRoute(spin)
	Total_Length = getRealTotaldistance(Route)
	elapsed_time = time.clock()-t0

	print("最短的路线是:{}".format(Route))
	print("最短距离{}".format(Total_Length))
	print("处理时间{}s".format(elapsed_time))

	plt.plot(LengthList)
	plt.show()
	D.draw_Route(POINT,matrix,Route)
