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

######## GLOBAL NONMYOPIC PATH #########
def global_evaluate(pf,cov,ax,ay,gridsize,grid_extent,k_mu=1.0, k_cov=1.0, k_range = 0.01,search_depth = 4,inner_radius = 100, outer_radius = 600):
    xx = np.linspace(0,1000, num=gridsize)
    yy = np.linspace(0,1000, num=gridsize)

    px = []
    py = []

    for x in xx:
        for y in yy:
            px.append(x)
            py.append(y)

    scores = []
    uncertainty_scores = []
    prediction_scores = []
    range_scores = []
    for gg in range(gridsize**2):
        ps = k_mu*pf[gg]

        k_bb = cov[gg,gg]
        k_sb = cov[:, gg]
        k_bs = k_sb.T
        invkb = 1/(k_bb+0.01)
        # UNCOMMENT ONE
        us = k_cov * (invkb * np.trace( np.outer(k_sb, k_bs)))
        # us = k_cov*cov[gg,gg] # Point uncertainty
        # us = k_cov * invkb * np.linalg.det(np.outer(k_sb, k_bs))

        #range_penalty = k_range * 0.0001*(np.sqrt((px[gg]-ax)**2+(py[gg]-ay)**2)-100)**2

        prediction_scores.append(ps)
        uncertainty_scores.append(us)
        #range_scores.append(-range_penalty)
        scores.append(ps+us) # potential uncertainty reduction

    origin = [ax,ay]
    no_segments = 10
    max_path_angle = 60*np.pi/180.0
    r = np.linspace(inner_radius,outer_radius, num = search_depth)
    theta =  np.linspace(0.0, 2.0*np.pi*(1.0-(1.0/no_segments)),num=no_segments)

    xgrid = []
    ygrid = []
    for rad in r:
        for t in theta:
            xgrid.append(ax + rad*np.cos(t))
            ygrid.append(ay + rad*np.sin(t))

    xgrid = np.array(xgrid)
    ygrid = np.array(ygrid)

    paths = []
    for i in range(search_depth):
        pp = []
        for k in range(no_segments**search_depth):
            pp.append(np.floor(k/(no_segments**(i)))%no_segments +i*no_segments)
        paths.append(pp)

    paths =  np.transpose(np.array(paths).astype(int))
    sorted_paths = []
    for path in paths:
        curr_origin = origin
        keep = True
        first = True
        for i in path:
            if not first:
                v1 = np.array([ygrid[i]-curr_origin[1],xgrid[i]-curr_origin[0]])
                v2 = np.array([curr_origin[1]-last_origin[1],curr_origin[0]-last_origin[0]])
                n_v1 = np.linalg.norm(v1)
                n_v2 = np.linalg.norm(v2)
                dp = np.dot(v1/n_v1,v2/n_v2)
                if dp < 0.0:
                    keep = False
                elif dp <= 1.0:
                    if abs(np.arccos(dp)) > max_path_angle:
                        keep = False
                else:
                    keep = False
            first = False
            last_origin = curr_origin
            curr_origin = [xgrid[i],ygrid[i]]
        if keep:
            sorted_paths.append(path)

    paths = np.array(sorted_paths)

    final_scores = []
    final_pscores = []
    final_uscores = []

    for ii, p in enumerate(paths):
        score = 0
        preds = 0
        uncs  = 0
        for pwp in p:
            x = xgrid[pwp]
            y = ygrid[pwp]
            if (x < 1000.0) and (x > 0.0) and (y > 0.0) and (y < 1000):
                xx = np.floor(x * gridsize / grid_extent[0])
                yy = np.floor(y * gridsize / grid_extent[1])
                gn = xx * gridsize + yy
                c = int(gn)
                # UNCOMMENT ONE
                # us = k_cov*cov[gg,gg] # Point uncertainty
                # us = k_cov * invkb * np.linalg.det(np.outer(k_sb, k_bs))
                score += prediction_scores[c] + uncertainty_scores[c]
                preds += prediction_scores[c]
                uncs  += uncertainty_scores[c]

        final_scores.append(preds)
        final_pscores.append(uncs)
        final_uscores.append(score)

    index = np.argmax(final_scores)
    print(index)
    return xgrid[paths[index][0]], ygrid[paths[index][0]], xgrid, ygrid, final_scores[index],final_uscores[index],final_pscores[index],#TODO scores, range_scores[index],xgrid[paths[index]], ygrid[paths[index]],xgrid,ygrid


auv = vehicle()
model = FlatGaussianRandomField(mu_0, length_scale,gridsize,[1000,1000],sigma2=sill,kernel="SE", nugget=nugget)
trackx = []
tracky = []
rmse = []
sigma_rmsen = []
sigma_rmsep = []
trcov =[]
adaption_time = []
scores = []
pscores = []
uscores = []
rscores = []
figno = 0

for t in range(mission_time):
    auv.update()
    if auv.state:
        vals, gno = segment(auv.measurements,auv.xmeasurements,auv.ymeasurements,gridsize,grid_extent)
        pf,cov = model.evaluate(vals,gno)

        eval = global_evaluate(pf,cov,auv.x,auv.y,gridsize,grid_extent,k_mu=0.01)
        auv.wp = eval[0:2]
        auv.measurements  = []
        auv.xmeasurements = []
        auv.ymeasurements =[]

        trackx.append(auv.x)
        tracky.append(auv.y)

        scores.append(eval[4])
        pscores.append(eval[6])
        uscores.append(eval[5])
        rscores.append(eval[8])

        se = (pf-clean_val)**2
        rmse.append(np.sqrt(np.mean(se)))
        sigma_rmsen.append(np.sqrt(np.mean(se))-np.sqrt(np.var(se)))
        sigma_rmsep.append(np.sqrt(np.mean(se))+np.sqrt(np.var(se)))
        adaption_time.append(t)

        trcov.append(np.trace(cov))

        # PLOTTING
        fig, ax = plt.subplots(3,2, figsize=(8, 9))
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

        ax[1,0].plot(adaption_time,pscores, label = "$\mu$ score")
        ax[1,0].plot(adaption_time,uscores, label = "$\Sigma^2$ score")
        ax[1,0].plot(adaption_time,rscores, label = "Range penalty")
        ax[1,0].plot(adaption_time,scores, label = "Path score")
        ax[1,0].legend()

        pathx = np.insert(eval[9],0,auv.x,axis=0)
        pathy = np.insert(eval[10],0,auv.y,axis=0)
        cbar = ax[1,1].scatter(eval[2],eval[3],c=eval[7])
        fig.colorbar(cbar,ax= ax[1,1])
        ax[1,1].scatter(eval[11],eval[12])
        ax[1,1].scatter(auv.x, auv.y, c="red", s=100)
        ax[1,1].scatter(eval[0],eval[1],c="red",s = 80, marker = "*")
        ax[1,1].set_ylim(0,1000)
        ax[1,1].set_xlim(0,1000)

        ax[2,0].plot(adaption_time,rmse,c="red")
        ax[2,0].fill_between(adaption_time,sigma_rmsep,y2=sigma_rmsen,color="red",alpha = 0.3)
        ax[2,0].scatter(adaption_time,rmse,c="red",s=40,marker = "s")

        ax[2,1].plot(adaption_time,trcov,c="red")
        ax[2,1].scatter(adaption_time,trcov,c="red",s=40,marker = "s")

        ax[0,0].set_title("Predictive mean, $\mu$")
        ax[0,1].set_title("Variance, $\Sigma^2$")
        ax[1,1].set_title("WPs and score",y=0.8, x= 0.7)
        ax[2,0].set_title("RMSE $\pm 1\sigma$",y=0.8, x= 0.7)
        ax[2,1].set_title("Trace($\Sigma^2$)",y=0.8, x= 0.8)

        plt.savefig("fig/example5/update_step" + str(int(figno)) + ".png",dpi = 300)
        plt.close()
        figno += 1

        print(str(int(100*t/mission_time)),"% done")
