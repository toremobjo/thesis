import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import plotly.express as px
import SpatialLogGP
from tempfile import TemporaryFile
from simulator_cf import getValue

class AUV:
    def __init__(self,x_init=-100.0,y_init=-200,update_period = 1,vno=0,no_vs = 4, grid_extent = [3000,3000,50],grid_size=[15,15,10],lscales = [600,600,3],sigma2 = 2.3,nugg = 0.5, timesat = 10000, orientation = 0.0):
        self.x = x_init
        self.y = y_init
        self.z = 0.0

        self.vehicle_no = vno
        self.no_vehicles = no_vs

        self.maxpitch = 20*np.pi/180
        self.speed= 1.5
        self.update_period = update_period

        self.avoidance_coeff = 300

        lat_0 = 78.93295
        lon_0 = 11.953366

        circ = 40075000.0
        self.lat = lat_0 + self.x*360.0/circ
        self.lon = lon_0 + self.y*360.0/(circ*np.cos(lat_0*np.pi/180.0))

        self.state = 0
        self.wp = [0,0,10]

        self.other_vs = []
        for i in range(no_vs):
            self.other_vs.append([-100000, -100000])

        self.grid_extent = grid_extent
        self.grid_size = grid_size
        self.lscales = lscales
        self.sigma2 = sigma2
        self.nugg = nugg
        self.timesat = timesat
        self.nplaces = self.grid_size[0]*self.grid_size[1]*self.grid_size[2]

        self.divelength = 2 * self.grid_extent[2] / np.tan(self.maxpitch) + 50.0
        self.dives = np.floor(self.grid_extent[0] / self.divelength)

        self.prior_wps = []

        self.dives = 1000
        n_rows = 5
        padding = 0.05 * self.grid_extent[0]

        wp_x = []
        wp_y = []

        for i in range(n_rows * 2):
            wp_x.append((np.floor((i + 1) / 2.0) % 2) * (self.grid_extent[0] - 2 * padding) + padding)
            wp_y.append(
                ((1.0 / (n_rows - 1)) * np.floor((i) / 2.0) % 2) * (self.grid_extent[1] - 2 * padding) + padding)

        for i in range((n_rows * 2) - 1):
            l = np.sqrt((wp_x[i] - wp_x[i + 1]) ** 2 + (wp_y[i] - wp_y[i + 1]) ** 2)

            if l > self.divelength:
                dives = np.floor(l / self.divelength)
                lx = (wp_x[i + 1] - wp_x[i]) / (2.0 * dives)
                ly = (wp_y[i + 1] - wp_y[i]) / (2.0 * dives)
                for j in range(2 * int(dives) + 1):
                    self.prior_wps.append([wp_x[i] + lx * j, wp_y[i] + ly * j, (j % 2) * self.grid_extent[2] + 0.01])

        self.prior_wps = self.prior_wps[0:18]

        self.init = False
        self.adapting = False
        self.prior = False
        self.update_iteration = 0
        self.wp_update_iteration = 0

        self.chlas = []
        self.df = pd.DataFrame(columns=["d", "c", "t"])
        self.df_to_send = pd.DataFrame(columns=["d", "c", "t"])

        self.gp = SpatialLogGP.GriddedLogGaussianProcess3D(self.lscales,
                                                           self.grid_size, self.grid_extent, self.sigma2,
                                                           time_sat=self.timesat, nugget=self.nugg)

        self.generatePaths()
        self.xl = []
        self.yl = []
        self.zl = []
        self.uil = []
        self.scorel = []
        self.pmeanl = []
        self.uncl = []

        self.ascending = False
        self.yoyo = True

    def get_xyz(self):
        return [self.x, self.y, self.z]

    def set_wp(self, wp):
        self.wp = wp
        self.update()

    def update(self):
        self.update_iteration += 1
        ddz = 2.0 * self.grid_extent[2] / self.grid_size[2]
        if self.state:
            self.x = self.x + (self.wp[0] - self.x) * self.speed / np.linalg.norm(
                np.array([self.x - self.wp[0], self.y - self.wp[1]])) * self.update_period
            self.y = self.y + (self.wp[1] - self.y) * self.speed / np.linalg.norm(
                np.array([self.x - self.wp[0], self.y - self.wp[1]])) * self.update_period

            # DIVE/ASCENT ENVELOPE WITHIN SAMPLINNG TO USE THE LARGE VERTICAL VARIANCE
            if self.yoyo:
                if self.ascending and abs(self.z - self.wp[2] + ddz) < 0.5:
                    self.ascending = False
                if (not self.ascending) and abs(self.z - self.wp[2] - ddz) < 0.5:
                    self.ascending = True

                if not self.ascending:
                    self.z = self.z + np.sign(self.wp[2] - self.z + ddz) * self.speed * np.sin(
                        self.maxpitch) * self.update_period
                else:
                    self.z = self.z + np.sign(self.wp[2] - self.z - ddz) * self.speed * np.sin(
                        self.maxpitch) * self.update_period

            else:
                if abs(self.wp[2] - self.z) > 0.5:
                    self.z = self.z + np.sign(self.wp[2] - self.z) * self.speed * np.sin(
                        self.maxpitch) * self.update_period

        self.z = max([self.z, 0.01])
        self.z = min([self.z, self.grid_extent[2] - 0.01])
        self.state = 5 < np.linalg.norm(np.array([self.x - self.wp[0], self.y - self.wp[1]]))

        cv = getValue(self.x, self.y, self.z, time=self.update_iteration * self.update_period, noise=True)
        self.chlas.append([self.x, self.y, self.z, self.update_iteration * self.update_period, cv])
        self.xl.append(self.x)
        self.yl.append(self.y)
        self.zl.append(-self.z)

        self.uil.append(self.update_iteration)

    def segment(self):
        if len(self.chlas) > 2:
            chlas = np.array(self.chlas)
            x = chlas.T[0]
            y = chlas.T[1]
            z = chlas.T[2]
            t = chlas.T[3]
            data = chlas.T[4]
            xx = np.floor(x * self.grid_size[0] / self.grid_extent[0])
            yy = np.floor(y * self.grid_size[1] / self.grid_extent[1])
            zz = np.floor(z * self.grid_size[2] / self.grid_extent[2])
            gn = xx * self.grid_size[1] * self.grid_size[2] + yy * self.grid_size[2] + zz
            c = gn.astype(int)
            df = pd.DataFrame({"d": data, "c": c, "t": t})

            c = []
            data = []
            t = []
            for i in range(0, self.nplaces):
                d = df[df["c"] == i]
                if len(d) > 0:
                    c.append(i)
                    data.append(d["d"].mean())
                    t.append(d["t"].mean())
            df = pd.DataFrame({"d": data, "c": c, "t": t})

            self.df = pd.concat([self.df, df], ignore_index=True)
            self.df.sort_values(by="t", inplace=True)
            self.df.drop_duplicates(subset="c", keep="last", inplace=True)
            self.df.sort_values(by="c", inplace=True)
            self.chlas = []

    def ingest_df(self, df):
        self.df = pd.concat([self.df, df], ignore_index=True)
        self.df.sort_values(by="t", inplace=True)
        self.df.drop_duplicates(subset="c", keep="last", inplace=True)
        self.df.sort_values(by="c", inplace=True)

    def generatePaths(self):
        r = 1.0 * np.sqrt(2) * self.grid_extent[0] / self.grid_size[0]
        ddz = 2.0 * self.grid_extent[2] / self.grid_size[2]
        dz = [-ddz, 0.0, ddz]
        theta = np.linspace(0.0, 2.0 * np.pi * (1.0 - (1.0 / 8)), 8)
        paths = []
        for t in theta:
            for zz in dz:
                base_node = np.array([r * np.cos(t), r * np.sin(t), zz])
                paths.append(np.array([base_node, base_node + np.array(
                    [r * np.cos(t - np.pi / 4.0), r * np.sin(t - np.pi / 4.0), zz - ddz])]))
                paths.append(np.array(
                    [base_node, base_node + np.array([r * np.cos(t - np.pi / 4.0), r * np.sin(t - np.pi / 4.0), zz])]))
                paths.append(np.array([base_node, base_node + np.array(
                    [r * np.cos(t - np.pi / 4.0), r * np.sin(t - np.pi / 4.0), zz + ddz])]))
                paths.append(np.array([base_node, base_node + np.array([r * np.cos(t), r * np.sin(t), zz - ddz])]))
                paths.append(np.array([base_node, base_node + np.array([r * np.cos(t), r * np.sin(t), zz])]))
                paths.append(np.array([base_node, base_node + np.array([r * np.cos(t), r * np.sin(t), zz + ddz])]))
                paths.append(np.array([base_node, base_node + np.array(
                    [r * np.cos(t + np.pi / 4.0), r * np.sin(t + np.pi / 4.0), zz - ddz])]))
                paths.append(np.array(
                    [base_node, base_node + np.array([r * np.cos(t + np.pi / 4.0), r * np.sin(t + np.pi / 4.0), zz])]))
                paths.append(np.array([base_node, base_node + np.array(
                    [r * np.cos(t + np.pi / 4.0), r * np.sin(t + np.pi / 4.0), zz + ddz])]))

        self.paths = np.array(paths)

    def adapt(self):
        pf, pc = self.gp.evaluate(self.df)

        scores = []
        wps = []
        lmp = []
        lavoid = []
        lunc = []
        pp = copy.deepcopy(self.paths)

        for path in pp:
            for i in range(2):
                path[i][0] += self.wp[0]
                path[i][1] += self.wp[1]
                path[i][2] += self.wp[2]

            if path[0][2] >= self.grid_extent[2]:
                path[0][2] = self.grid_extent[2] - 0.01

            if path[0][2] < 0.0:
                path[0][2] = 0.01

            if path[0][0] > 0.0 and path[0][0] < self.grid_extent[0] and path[0][1] > 0.0 and path[0][1] < \
                    self.grid_extent[1] and path[0][2] > 0.0 and path[0][2] < self.grid_extent[2]:
                if path[1][0] > 0.0 and path[1][0] < self.grid_extent[0] and path[1][1] > 0.0 and path[1][1] < \
                        self.grid_extent[1] and path[1][2] > 0.0 and path[1][2] < self.grid_extent[2]:
                    xx = np.floor(path[0][0] * self.grid_size[0] / self.grid_extent[0])
                    yy = np.floor(path[0][1] * self.grid_size[1] / self.grid_extent[1])
                    zz = np.floor(path[0][2] * self.grid_size[2] / self.grid_extent[2])
                    gn = xx * self.grid_size[1] * self.grid_size[2] + yy * self.grid_size[2] + zz
                    c_inner = gn.astype(int)

                    xx = np.floor(path[1][0] * self.grid_size[0] / self.grid_extent[0])
                    yy = np.floor(path[1][1] * self.grid_size[1] / self.grid_extent[1])
                    zz = np.floor(path[1][2] * self.grid_size[2] / self.grid_extent[2])
                    gn = xx * self.grid_size[1] * self.grid_size[2] + yy * self.grid_size[2] + zz
                    c_outer = gn.astype(int)
                    avoidance_score = 300.0

                    for i, av in enumerate(self.other_vs):
                        if not i == self.vehicle_no:
                            avoidance_score -= self.avoidance_coeff ** 2 / (
                                        (path[0][0] - av[0]) ** 2 + (path[0][1] - av[1]) ** 2)

                    values = []
                    ss = pc[c_inner, c_inner] + pc[c_outer, c_outer]
                    values.append(pf[c_inner] + pf[c_outer])
                    lmp.append(pf[c_inner] + pf[c_outer])
                    lunc.append(pc[c_inner, c_inner] + pc[c_outer, c_outer])
                    lavoid.append(avoidance_score)
                    scores.append(ss + pf[c_inner] + pf[c_outer] + avoidance_score)
                    wps.append([path[0][0], path[0][1], path[0][2]])

        index = np.argmax(scores)

        # Log scores, predictive mean, uncertainty, avoidance score, update time, predicted field, and uncertainties
        self.scorel.append(
            [scores[index], lmp[index], lunc[index], lavoid[index], self.update_iteration * self.update_period, pf,
             np.diag(pc)])
        wp = wps[index]
        return wp

    def ingest_pos(self, pos, vno):
        self.other_vs[vno] = [pos[0], pos[1]]

    def get_state(self):
        return self.state