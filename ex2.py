#coding:utf-8
import time
import math
import numpy as np
import os
import random
import copy
import matplotlib.pyplot as plt


def getSpinConfig():
    def spin_config_at_a_time_in_a_TROTTER_DIM(tag):
        config = list(-np.ones(NCITY, dtype = np.int))#生成一维数组，大小为ncity，数据为-1，类型为int
        config[tag] = 1#改变tag位置上的数据为1
        return config#返回一个大小为ncity，tag位数为1，其余为-1的一维数组
        
    def spin_config_in_a_TROTTER_DIM(tag):#tag是一维数组，大小为从1到ncity
        spin = []
        spin.append(config_at_init_time)
        for i in range(TOTAL_TIME-1):
            spin.append(list(spin_config_at_a_time_in_a_TROTTER_DIM(tag[i])))
        return spin

    spin = []
    for i in range(TROTTER_DIM):  #TROTTER_DIM10
        tag = np.arange(1,NCITY)#生成1-ncity的一维数组
        np.random.shuffle(tag)#将数组顺序打乱
        spin.append(spin_config_in_a_TROTTER_DIM(tag))
    return spin


# 选择Trotter的尺寸是最短距离，输出该时刻的路线
def getBestRoute(d_matrix,config):
    length = []
    for i in range(TROTTER_DIM):
        route = []
        for j in range(TOTAL_TIME):
            route.append(config[i][j].index(1))
        length.append(getTotaldistance(d_matrix,route))

    min_Tro_dim = np.argmin(length)   #返回数组最小值的位置
    Best_Route = []
    for i in range(TOTAL_TIME):
        Best_Route.append(config[min_Tro_dim][i].index(1))
    return Best_Route


##统计距离
def getTotaldistance(d_matrix,route):
    Total_distance = 0
    for i in range(TOTAL_TIME):
        Total_distance += distance(d_matrix,route[i],route[(i+1)%TOTAL_TIME])/max_distance
    return Total_distance


## 真实距离
def getRealTotaldistance(d_matrix,route):
    Total_distance = 0
    for i in range(TOTAL_TIME):
        Total_distance += distance(d_matrix,route[i],route[(i+1)%TOTAL_TIME])
    return Total_distance


## 量子蒙特卡罗步骤
def QMC_move(d_matrix,config, ann_para):
    # 两个不同的时间a,b选
    c = np.random.randint(0,TROTTER_DIM) #TROTTER_DIM=10
    a_ = list(range(1,TOTAL_TIME))####添加了list才能用remove
    a = np.random.choice(a_)
    a_.remove(a)#排除已经选择了的a
    b = np.random.choice(a_)
    # 在一些＃特洛塔数字c，的时间的，城市P中B
    p = config[c][a].index(1)
    q = config[c][b].index(1)
    # 初始化的能量差的值  
    delta_cost = delta_costc = delta_costq_1 = delta_costq_2 = delta_costq_3 = delta_costq_4 = 0

    # （7）来回翻转自旋的能量差的第一项 
    for j in range(NCITY):
        l_p_j = distance(d_matrix,p,j)/max_distance
        l_q_j = distance(d_matrix,q,j)/max_distance
        delta_costc += 2*(-l_p_j*config[c][a][p] - l_q_j*config[c][a][q])*(config[c][a-1][j]+config[c][(a+1)%TOTAL_TIME][j])+2*(-l_p_j*config[c][b][p] - l_q_j*config[c][b][q])*(config[c][b-1][j]+config[c][(b+1)%TOTAL_TIME][j])
	#delta_costc=2*sum((-lpj + lqj)*(config[c][][j] - config[c][(a+1)%TOTAL_TIME]))

    # 之前和的能量差之后翻转自旋为等式（7）的第二项  
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



def creat_dist_matrix(P,N):
	dist_matrix =[]
	row=[]
	#初始化距离矩阵
	for i in range(N):
		for j in range(N):
			row.append(65535)
		dist_matrix.append(row)
		row=[]
	
	#构建非全连同图矩阵
	for i in range(N):
		length=len(LINK[i])
		for j in range(length-1):
			pos=POINTNAME.index(LINK[i][j+1])
			dist=math.sqrt((float(P[i][0])-float(P[pos][0]))**2 + (float(P[i][1])-float(P[pos][1]))**2)
			dist=round(dist,2)
			dist_matrix[i][pos]=dist

	return dist_matrix
       

def draw_Line(P,D_Matrix,R):
	#绘制道路网
	for i in range(len(D_Matrix)):
		for j in range(i,len(D_Matrix)):
			if D_Matrix[i][j]!=65535:
				i+j
				plt.plot([P[i][0],P[j][0]],[P[i][1],P[j][1]],linewidth=12.0,zorder=2,c='b')

	#绘制坐标点
	for i in range(len(P)):
		plt.scatter(P[i][0], P[i][1],s=120,zorder=3, color='b')
		plt.text(P[i][0], P[i][1],P[i][2],size=10,ha="center", va="center", bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
	#绘制路线
	for i in range(len(R)):
		plt.plot([P[i][0],P[(i+1)%len(R)][0]],[P[i][1],P[(i+1)%len(R)][1]],linewidth=3.0,zorder=3,c='r')
		plt.text((P[i][0]+P[(i+1)%len(R)][0])/2,(P[i][1]+P[(i+1)%len(R)][1])/2, str(i+1), fontsize=16, rotation_mode='anchor')
		
	plt.grid(True,ls='--')
	plt.show()

def draw_route(R,P):

	for i in range(len(R)):
		plt.plot([P[i][0],P[(i+1)%len(R)][0]],[P[i][1],P[(i+1)%len(R)][1]],c='r')

	for i in range(len(P)):
		plt.scatter(P[i][0], P[i][1],s=120,zorder=3, color='b')
		plt.text(P[i][0], P[i][1],P[i][2],size=10,ha="center", va="center", bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
 
	plt.grid(True,ls='--')
	plt.show()



def Floyd(G):
	#节点数
	Length=len(G)
	#创建A，Path矩阵
	A=np.zeros((Length,Length),dtype=float)
	Path=np.zeros((Length,Length),dtype=int)

	#初始化矩阵A，Path
	for i in range(Length):
		for j in range(Length):
			A[i][j]=G[i][j]
			Path[i][j]=-1

	for k in range(Length):
		for i in range(Length):
			for j in range(Length):
				if A[i][j]>(A[i][k]+A[k][j]):
					A[i][j]=A[i][k]+A[k][j]
					Path[i][j]=k
	
	'''for i in range(Length):
		print(A[i])

	for i in range(Length):
		print(Path[i])
	'''

	return A

## 2城市之间距离
def distance(d_matrix,i,j):
	return d_matrix[i][j]

# 参数的输入
#TROTTER_DIM = int(input("Trotter dimension: "))
#ANN_PARA =  float(input("initial annealing parameter: "))
#ANN_STEP = int(input("Annealing Step: "))
#MC_STEP = int(input("MC step: "))
#BETA = float(input("inverse Temperature: "))

TROTTER_DIM = 10
ANN_PARA =  1.0
ANN_STEP = 400
MC_STEP = 2320
BETA = 37
REDUC_PARA = 0.99

"""
获取城市（城市号和x，y坐标）的数据
"""

#读取点坐标
f = open('./ex2.txt').read().split("\n")
POINT = []
POINTNAME=[]
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


for i in range(NCITY):
	POINTNAME.append(POINT[i][2])
	for j in range(2):
		POINT[i][j] = float(POINT[i][j])


"""
量子退火模拟
"""
if __name__ == '__main__':

	# 2城市间的距离的最大np.zeros值
	matrix=creat_dist_matrix(POINT,NCITY)
	#matrix_origin=copy.deepcopy(matrix)
	d_matrix=Floyd(matrix)
	max_distance =0
    #求最大值
	for i in range(NCITY):
		for j in range(NCITY):
			if max_distance <= distance(d_matrix,i, j):
				max_distance = distance(d_matrix,i, j)
    
	config_at_init_time = list(-np.ones(NCITY,dtype=np.int))
    
	config_at_init_time[0] = 1

	print("start...")
	t0 = time.clock()

	np.random.seed(100)
	spin = getSpinConfig()


   ################################################################## print(spin)
	LengthList = []
	for t in range(ANN_STEP):
		for i in range(MC_STEP):
			con = QMC_move(d_matrix,spin, ANN_PARA)
			rou = getBestRoute(d_matrix,con)
			length = getRealTotaldistance(d_matrix,rou)
		LengthList.append(length)
        #print("Step:{}, Annealing Parameter:{}, length:{}".format(t+1,ANN_PARA, length))
		ANN_PARA *= REDUC_PARA



	Route = getBestRoute(d_matrix,spin)
	Total_Length = getRealTotaldistance(d_matrix,Route)
	elapsed_time = time.clock()-t0

	print("最短的路线是:{}".format(Route))
	print("最短距离{}".format(Total_Length))
	print("处理时间{}s".format(elapsed_time))

	plt.plot(LengthList)
	plt.show()
	#绘制最终线路图
	draw_Line(POINT,matrix,Route)
