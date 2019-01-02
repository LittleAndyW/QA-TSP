import math
import numpy as np


def creat_dist_matrix(P,LINK):
	dist_matrix =[]
	row=[]
	N=len(P)
	POINTNAME=[]   #存储点名称
	for i in range(N):
		POINTNAME.append(P[i][2])
		for j in range(2):
			P[i][j] = float(P[i][j])
	
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


def NetOpt():
	#creat_dist_matrix()
	print()

def Floyd(G):
	#节点数
	A_Path=[]   #存储距离矩阵与路径矩阵
	Length=len(G)
	#创建A，Path矩阵
	A=np.zeros((Length,Length),dtype=float)
	Path=np.zeros((Length,Length),dtype=int)


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
	A_Path.append(A)
	A_Path.append(Path)

	return A_Path  #返回距离矩阵与路径矩阵

def getPath(u,v, Path,route):
	if Path[u][v] ==-1:
		route.append([u,v])
	else:
		mid=path[u][v]
		getPath(u,mid, Path,route)
		getPath(mid,v, Path,route)

