from utils import vehicle, segment, getTempVal, FlatGaussianRandomField
import numpy as np
import matplotlib.pyplot as plt

gridsize = 50
xx = np.linspace(0,1000, num=gridsize)
yy = np.linspace(0,1000, num=gridsize)


val = []
clean_val = []
xval = []
yval = []

for x in xx:
    for y in yy:
        val.append(getTempVal(x,y))
        clean_val.append(getTempVal(x,y,noise=False))
        xval.append(x)
        yval.append(y)


val = np.array(val)
clean_val = np.array(clean_val)

fig, ax = plt.subplots(2,2,figsize = (10,8))

aaa = ax[0,0].contourf(np.reshape(xval,(gridsize,gridsize)),np.reshape(yval,(gridsize,gridsize)),np.reshape(val,(gridsize,gridsize)),10)
ax[0,0].set_title("Measurement field with noise")
bbb = ax[1,0].contourf(np.reshape(xval,(gridsize,gridsize)),np.reshape(yval,(gridsize,gridsize)),np.reshape(clean_val,(gridsize,gridsize)),10)
ax[1,0].set_title("Underlying field")

corners = [[50,50],[950,50],[50,950],[950,950]]
cornervalues  =[]
for corner in corners:
    cornervalues.append(getTempVal(corner[0],corner[1]))
corners = np.array(corners)

ax[0,0].scatter(corners.T[0],corners.T[1],c="red", s = 100)

X = np.array([np.ones(len(corners)), corners.T[0], corners.T[1]]).T
b = np.linalg.lstsq(X, cornervalues, rcond=-1)[0]
mu = b[0]*np.ones(gridsize*gridsize) + b[1]*np.array(xval) + b[2]*np.array(yval)
mu_0 = mu
mu = np.reshape(mu,(gridsize,gridsize))

ccc = ax[0,1].contourf(np.reshape(xval,(gridsize,gridsize)),np.reshape(yval,(gridsize,gridsize)),mu,10)
ax[0,1].set_title("Prior from measurements")

ddd = ax[1,1].contourf(np.reshape(xval,(gridsize,gridsize)),np.reshape(yval,(gridsize,gridsize)),mu-np.reshape(val,(gridsize,gridsize)),10)
ax[1,1].set_title("Error in prior, $\mu-x$")

fig.colorbar(aaa,ax=ax[0,0])
fig.colorbar(bbb,ax=ax[1,0])
fig.colorbar(ccc,ax=ax[0,1])
fig.colorbar(ddd,ax=ax[1,1])
plt.savefig("fig/example0/field.png",dpi=300)
plt.show()
