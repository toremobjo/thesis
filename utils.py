#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as scip

class FlatGaussianRandomField:
    def __init__(self,prior,lscales,grid_size,grid_extent,sigma2 = 0.1,kernel="SE", nugget=0.01):

        self.ndim = 2                       # number of dimensions
        self.lscales = lscales              # length scale for each ndim
        self.grid_size = grid_size          # number of nodes in grid for each of ndim
        self.grid_extent = grid_extent      # extent of each grid dimension in meters
        self.kernel = kernel                # Wich kernel to use
        self.nugget = nugget                # Nugget effect variance
        self.sigma2 = sigma2                # Sigma squared, variance of all data
        self.nogrid = grid_size*grid_size
        self.mu = prior

        e = []
        g = []
        for i in range(0,self.ndim):
            e.append(np.linspace(0,grid_extent[i],num=grid_size+1))
            g.append(np.linspace(grid_extent[i]/(grid_size*2),grid_extent[i]-grid_extent[i]/(grid_size*2),num=grid_size))

        self.grid = []
        self.gridno = []
        i=0
        for x in g[0]:
            for y in g[1]:
                self.grid.append([x,y])
                self.gridno.append(i)
                i+=1

        self.F = np.zeros(len(self.grid))  # Design matrix, F_ikj = 1 for all cells containing measurements
        self.grid = np.array(self.grid)
        self.gridno = np.array(self.gridno)

        #Generate covariance matrix
        if self.kernel == "SE":
            self.cov = self.get_se_cov(self.lscales,self.sigma2,self.grid)
            #self.cov = self.getcov(self.grid,self.grid,[self.sigma2,self.lscales])
        else:
            raise Exception('Implement your own GD kernel')
        print("GP Initialized")

    def get_se_cov(self,lscales,sigma2,sites):
        xx = sites.T[0]/lscales
        yy = sites.T[1]/lscales
        xa = xb = np.array((xx,yy)).T
        sqnorm = -0.5 * scip.distance.cdist(xa,xb,"sqeuclidean")
        cov =  sigma2*np.exp(sqnorm)
        return cov

    def getcov(self,d_sites, p_sites, par):
        sig2 = par[0]
        crange = par[1]

        h = -0.5 * scip.distance.cdist(d_sites / crange, p_sites / crange, 'sqeuclidean')
        # sqnorm = -0.5 * scip.distance.cdist(xa, xb, "sqeuclidean")
        cov = sig2 * np.exp(h)
        return cov

    def update_f(self,grid_no):
        self.F = np.zeros(len(self.grid),dtype=bool)
        self.F[grid_no]=True

    def evaluate(self,data,grid_no):
        self.update_f(grid_no)
        d_d = data - self.mu[grid_no]

        tau = np.diag(self.nugget * np.ones(sum(self.F)))
        k_bb = self.cov[self.F].T[self.F] + tau
        k_sb = self.cov[:, self.F]
        k_bs = k_sb.T
        k_ss = self.cov

        invkb = np.linalg.inv(k_bb)
        pred_field = k_sb @ invkb @ d_d + self.mu
        resulting_cov = k_ss - k_sb @ invkb @ k_bs

        self.mu = pred_field
        self.cov = resulting_cov

        return pred_field,resulting_cov


def getTempVal(x,y,noise = True):
    tmean = 8.3435342
    tsin = -0.9 * np.sin(x/1000.0 + np.pi/6) - 0.6 * np.sin(y/1000.0 - np.pi/6)
    tplume = 1.2*np.exp(((-0.2*(x-750)**2)+(-(y-700)**2))/9000)
    if noise:
        return tmean + tsin + tplume + np.random.normal(0.0,0.1) # nugget = 0.1
    else:
        return tmean + tsin + tplume

def segment(values, x, y, gridsize, grid_extent):
    c = []
    for i, v in enumerate(values):
        if (x[i] < 1000.0) and (x[i] > 0.0) and (y[i] > 0.0) and (y[i] < 1000):
            xx = np.floor(x[i] * gridsize / grid_extent[0])
            yy = np.floor(y[i] * gridsize / grid_extent[1])
            gn = xx * gridsize + yy
            c.append(int(gn))

    val = []
    grid_no = []

    for cc in range(gridsize*gridsize):
        try:
            index = c.index(cc)
            if index:
                val.append(np.mean(values[index]))
                grid_no.append(int(cc))
        except:
            pass
    return val, grid_no


class vehicle:
    def __init__(self):
        self.x = 10
        self.y = 10

        self.measurements  = []
        self.xmeasurements = []
        self.ymeasurements = []

        self.speed = 1.5

        self.wp = [10,10]
        self.state = 0

    def update(self):
        self.state = 5 > np.linalg.norm(np.array([self.x-self.wp[0],self.y-self.wp[1]]))

        heading = np.arctan2(self.wp[1]-self.y,self.wp[0]-self.x)
        self.x += self.speed*np.cos(heading)
        self.y += self.speed*np.sin(heading)

        self.measurements.append(getTempVal(self.x,self.y))
        self.xmeasurements.append(self.x)
        self.ymeasurements.append(self.y)
