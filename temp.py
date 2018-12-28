import matplotlib.pyplot as plt
import numpy as np

# Plot diagonal line (45 degrees)
h = plt.plot(np.arange(0, 100), np.arange(0, 100))

# set limits so that it no longer looks on screen to be 45 degrees
plt.xlim([-100, 200])

# Locations to plot text
l1 = np.array((10, 10))
l2 = np.array((50, 50))

# Rotate angle
angle = 0.45
trans_angle = plt.gca().transData.transform_angles(np.array((45,)),
                                                   l2.reshape((1, 2)))[0]

# Plot text
th1 = plt.text(l1[0], l1[1], '1', fontsize=16,
               rotation=angle, rotation_mode='anchor')
th2 = plt.text(l2[0], l2[1], 'text rotated correctly', fontsize=16,
               rotation=trans_angle, rotation_mode='anchor')

plt.show()	

'''
np.random.seed(19680801)
data = np.random.randn(2, 100)

fig, axs = plt.subplots(1, 1, figsize=(5, 5))

axs.hist2d(data[0], data[1])

plt.show()





import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)


X = 10*np.random.rand(11, 11)

fig, ax = plt.subplots()
ax.imshow(X, interpolation='nearest')

numrows, numcols = X.shape


def format_coord(x, y):
    col = int(x + 0.5)
    row = int(y + 0.5)
    if col >= 0 and col < numcols and row >= 0 and row < numrows:
        z = X[row, col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f' % (x, y)

ax.format_coord = format_coord
plt.show()'''
