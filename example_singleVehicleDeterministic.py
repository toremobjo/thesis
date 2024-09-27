import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import SpatialLogGP
from AUVDET import AUV
from simulator_cf import getValue

# GP parameters
# Extent of grid in x,y, and z-directions
grid_e = [3000, 3000, 50]
# Number of grid cells in x, y, and z-directions
grid_s = [15, 15, 10]
# Spatial length scales in x, y and z-directions
length_scales = [600,600,3]
# De-correlation time
time_scale = 10000
# Orientation of operational volume
orientation_g = -45.0*np.pi/180.0
ndim = len(grid_e)
# Scale and nugget variance for log(x), change the model in AUV.py for another model.
sigma_squared  = 2.3
nugget = 0.7

# One iteration = one second, running here for 3 hours
run_iterations = 30915


# Initiate 4 AUVs:; Harald, Fritdjof, Roald, and Thor
nv = 1
h = AUV(vno=0,no_vs=4,grid_extent=grid_e,grid_size=grid_s,lscales=length_scales,orientation=orientation_g,timesat=time_scale,sigma2=sigma_squared,nugg=nugget)
f = AUV(x_init=200,y_init=100,vno=1,no_vs=4,grid_extent=grid_e,grid_size=grid_s,lscales=length_scales,orientation=orientation_g,timesat=time_scale,sigma2=sigma_squared,nugg=nugget)
r = AUV(x_init=1000,y_init=-500,vno=2,no_vs=4,grid_extent=grid_e,grid_size=grid_s,lscales=length_scales,orientation=orientation_g,timesat=time_scale,sigma2=sigma_squared,nugg=nugget)
t = AUV(x_init=0,y_init=0,vno=3,no_vs=4,grid_extent=grid_e,grid_size=grid_s,lscales=length_scales,orientation=orientation_g,timesat=time_scale,sigma2=sigma_squared,nugg=nugget)
auvs = [h,f,r,t]
auvs= auvs[:nv]

x = []
y = []
z = []
t = []

for run in range(run_iterations):
    for i, ag in enumerate(auvs):
        if not ag.get_state():
            ag.wp_update_iteration += 1
            print(int(100*ag.update_iteration/run_iterations), "Percent done")
            if not ag.init:
                ag.init = True
                ag.prior = True
                ag.set_wp(ag.prior_wps[0])

            elif ag.prior and ag.wp_update_iteration<len(ag.prior_wps):
                ag.set_wp(ag.prior_wps[ag.wp_update_iteration])

            elif ag.prior and ag.wp_update_iteration == len(ag.prior_wps):

                ag.prior = False
                ag.adapting = True
                ag.segment()
                df = ag.df
                for j in range(nv):
                    if auvs[j] != ag:
                        auvs[j].ingest_df(df)
                        auvs[j].ingest_pos(ag.get_xyz(),ag.vehicle_no)
                break
                ag.set_wp(ag.adapt())
            elif ag.adapting:
                break
                ag.segment()
                df = ag.df
                for j in range(nv):
                    if auvs[j] != ag:
                        auvs[j].ingest_pos(ag.get_xyz(), ag.vehicle_no)
                        auvs[j].ingest_df(df)
                ag.set_wp(ag.adapt())

        ag.update()
        xx = ag.get_xyz()
        if i==0:
            x.append(xx[0])
            y.append(xx[1])
            z.append(xx[2])
            t.append(run)

df = pd.DataFrame(columns=["x", "y", "z","auv"])
for i,ag in enumerate(auvs):
    tdf = pd.DataFrame({"x":ag.xl,"y":ag.yl,"z":ag.zl,"auv":str(i)})
    df = pd.concat([df,tdf])

fig = px.line_3d(df, x="x", y="y", z="z", color='auv')
fig.show()
fig.write_image("fig/NMsinglevehicleExample/auv_paths.png")

g = []
for i in range(0, ndim):
    g.append(np.linspace(grid_e[i] / (grid_s[i] * 2), grid_e[i] - grid_e[i] / (grid_s[i] * 2), num=grid_s[i]))

grid = []
for x in g[0]:
    for y in g[1]:
        for z in g[2]:
            grid.append([x, y, z])
grid = np.array(grid)

# Plot predictive mean
pf, pc = auvs[0].gp.evaluate(auvs[0].df)
fig = go.Figure(data=[go.Volume(
        x=grid.T[0],
        y=grid.T[1],
        z=-grid.T[2],
        value=np.log(pf),
        isomin=min(np.log(pf)),
        isomax=max(np.log(pf)),
        opacity=0.5,  # needs to be small to see through all surfaces
        surface_count=20,  # needs to be a large number for good volume rendering
        colorscale='Viridis',
        caps=dict(x_show=False, y_show=False, z_show=False),
    )])
fig.show()
fig.write_image("fig/NMsinglevehicleExample/predictions.png")


#plot uncertainty
fig = go.Figure(data=[go.Volume(
        x=grid.T[0],
        y=grid.T[1],
        z=-grid.T[2],
        value=np.log(np.diag(pc)),
        isomin=min(np.log(np.diag(pc))),
        isomax=max(np.log(np.diag(pc))),
        opacity=0.5,  # needs to be small to see through all surfaces
        surface_count=20,  # needs to be a large number for good volume rendering
        colorscale='Inferno',
        caps=dict(x_show=False, y_show=False, z_show=False),
    )])
fig.show()
fig.write_image("fig/NMsinglevehicleExample/variance.png")


ss = np.array(auvs[0].scorel)
plt.plot(ss.T[4],ss.T[0],label = "Total")
plt.plot(ss.T[4],ss.T[1],label = "predictive mean")
plt.plot(ss.T[4],ss.T[2],label = "unc")
plt.plot(ss.T[4],ss.T[3],label = "avoidance")
plt.legend()
plt.savefig("fig/NMsinglevehicleExample/pathscores_auv1.png",dpi = 300)
plt.close()


for ii in range(nv):
    ss = np.array(auvs[ii].scorel)

    rmse = []
    times = []
    for i, log in enumerate(ss):
        sim = []

        for j, g in enumerate(grid):
            sim.append((getValue(g[0],g[1],g[2],time=log[4],noise=False)-log[5][j])**2)
        rmse.append(np.sqrt(np.mean(sim)))
        times.append(log[4])

    plt.plot(times,rmse)
    with open('log2.npy', 'wb') as f:
        np.save(f, ss)
grid = np.array(grid)
plt.savefig("fig/NMsinglevehicleExample/rmse.png",dpi = 300)
plt.close()
