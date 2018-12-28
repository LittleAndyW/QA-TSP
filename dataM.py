# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:07:28 2018

@author: admins
"""

import csv
import numpy as np
from numpy import mat
import matplotlib.pyplot as plt

cities=[]
citydm=[]
temp=[]
i=0
#存储坐标
X=[]
Y=[]
temp_x=[]
temp_y=[]


#读取数据\
"""
csv_file=csv.reader(open('./20170105.csv'))
#生成列表
for city in csv_file:
    cities.append(city)
    #起点城市并不完全包含终点城市
    temp.append(city[2])      #起点城市
    temp.append(city[5])      #起点城市
    
    #记录起始点坐标
    temp_x.append(float(city[12]))
    temp_y.append(float(city[13]))
    temp_x.append(float(city[14]))
    temp_y.append(float(city[15]))
    
    i+=1
    """
    
#读取csv
f=open("./data.txt",'r', encoding='UTF-8') 
line=f.readline()
while line:
    cities.append(list(line.strip('\n').split(',')))
    
    line=f.readline()
f.close()

#生成列表
for city in cities:
    #起点城市并不完全包含终点城市
    temp.append(city[2])      #起点城市
    temp.append(city[5])      #起点城市
    
    #记录起始点坐标
    temp_x.append(float(city[12]))
    temp_y.append(float(city[13]))
    temp_x.append(float(city[14]))
    temp_y.append(float(city[15]))
    
    i+=1
#利用set去除重复的名称 并按照csv中的顺序进行存储
citydm=list(set(temp))
citydm.sort(key=temp.index)

#生成距离矩阵

print(len(citydm))

city_mat=np.zeros((len(citydm),len(citydm)),dtype=np.float64)
for i in range(0,len(citydm)):
    for j in range(0,len(citydm)):
        city_mat[i][j]=float('inf')

city_len=len(cities)
for k in range(0,city_len):
    idx_start=citydm.index(cities[k][2])
    idx_end=citydm.index(cities[k][5])
    city_mat[idx_start][idx_end]=cities[k][16]



#绘制各个城市的点位图



    
    
X=list(set(temp_x))
X.sort(key=temp_x.index)

Y=list(set(temp_y))
Y.sort(key=temp_y.index)


#plt.figure(figsize=(16,12))
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.rcParams['image.cmap'] = 'gray'

plt.plot(X,Y,'r-o')    #坐标点画直线
plt.scatter(X,Y,s=5,marker='<')   #散点图
plt.show()





#print(len(citydm[0]))

