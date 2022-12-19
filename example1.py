#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as scip
from utils import vehicle, segment, getTempVal, FlatGaussianRandomField

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

clean_val = np.array(clean_val)

val = np.array(val)


#Make prior from corner measurements
corners = [[50,50],[950,50],[50,950],[950,950]]
cornervalues  =[]
for corner in corners:
    cornervalues.append(getTempVal(corner[0],corner[1]))
corners = np.array(corners)


X = np.array([np.ones(len(corners)), corners.T[0], corners.T[1]]).T
b = np.linalg.lstsq(X, cornervalues, rcond=-1)[0]
mu = b[0]*np.ones(gridsize*gridsize) + b[1]*np.array(xval) + b[2]*np.array(yval)
mu_0 = mu
mu = np.reshape(mu,(gridsize,gridsize))

mission_time = 5000
length_scale = 300.0
grid_extent = [1000,1000]
nugget = 0.01
sill = np.var(val)

######## RANDOM PATH #########
auv = vehicle()
model = FlatGaussianRandomField(mu_0, length_scale,gridsize,[1000,1000],sigma2=sill,kernel="SE", nugget=nugget)
trackx = []
tracky = []
rmse = []
sigma_rmsen = []
sigma_rmsep = []
trcov =[]
adaption_time = []
figno = 0

for t in range(mission_time):
    auv.update()
    if auv.state:
        vals, gno = segment(auv.measurements,auv.xmeasurements,auv.ymeasurements,gridsize,grid_extent)
        pf,cov = model.evaluate(vals,gno)

        auv.wp = [np.random.randint(0,1000),np.random.randint(0,1000)]
        auv.measurements  = []
        auv.xmeasurements = []
        auv.ymeasurements =[]

        trackx.append(auv.x)
        tracky.append(auv.y)

        se = (pf-clean_val)**2
        rmse.append(np.sqrt(np.mean(se)))
        sigma_rmsen.append(np.sqrt(np.mean(se))-np.sqrt(np.var(se)))
        sigma_rmsep.append(np.sqrt(np.mean(se))+np.sqrt(np.var(se)))
        adaption_time.append(t)

        trcov.append(np.trace(cov))

        # PLOTTING
        fig, ax = plt.subplots(2,2, figsize=(8, 6))
        scp = ax[0,0].scatter(xval,yval,c=pf)
        fig.colorbar(scp,ax=ax[0,0])
        ax[0,0].scatter(auv.x,auv.y,c="red",s=100)
        ax[0,0].scatter(trackx,tracky,c="red",s=40,marker = "s")
        ax[0,0].plot(trackx, tracky, c="red")

        scc = ax[0,1].scatter(xval, yval, c=np.diag(cov),cmap="plasma")
        fig.colorbar(scc,ax=ax[0,1])
        ax[0,1].scatter(auv.x, auv.y, c="red", s=100)
        ax[0,1].scatter(trackx,tracky,c="red",s=40,marker = "s")
        ax[0,1].plot(trackx, tracky, c="red")

        ax[1,0].plot(adaption_time,rmse,c="red")
        ax[1,0].fill_between(adaption_time,sigma_rmsep,y2=sigma_rmsen,color="red",alpha = 0.3)
        ax[1,0].scatter(adaption_time,rmse,c="red",s=40,marker = "s")

        ax[1,1].plot(adaption_time,trcov,c="red")
        ax[1,1].scatter(adaption_time,trcov,c="red",s=40,marker = "s")

        ax[0,0].set_title("Predictive mean, $\mu$")
        ax[0,1].set_title("Variance, $\Sigma^2$")
        ax[1,0].set_title("RMSE($\mu-y$) $\pm 1\sigma$",y=0.8, x= 0.7)
        ax[1,1].set_title("Trace($\Sigma^2$)",y=0.8, x= 0.8)

        plt.savefig("fig/example1/update_step" + str(int(figno)) + ".png",dpi = 300)
        plt.close()
        figno += 1

        print(str(int(100*t/mission_time)),"% done")
