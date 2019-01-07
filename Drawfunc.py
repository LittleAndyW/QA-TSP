import matplotlib.pyplot as plt


def draw_Route(P,D_Matrix,R):
	#绘制道路网
	for i in range(len(D_Matrix)):
		for j in range(i,len(D_Matrix)):
			if D_Matrix[i][j]!=65535:
				plt.plot([P[i][0],P[j][0]],[P[i][1],P[j][1]],linewidth=8.0,zorder=2,c='b')

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


#构造路网
def create_Route(P,D_Matrix):
	#绘制道路网 
	for i in range(len(D_Matrix)):
		for j in range(i,len(D_Matrix)):
			if D_Matrix[i][j]!=65535:
				plt.plot([P[i][0],P[j][0]],[P[i][1],P[j][1]],linewidth=8.0,zorder=2,c='b')

	#绘制坐标点
	for i in range(len(P)):
		plt.scatter(P[i][0], P[i][1],s=120,zorder=3, color='b')
		plt.text(P[i][0], P[i][1],P[i][2],size=10,ha="center", va="center", bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
	#绘制路线

		
	plt.grid(True,ls='--')
	plt.show()

