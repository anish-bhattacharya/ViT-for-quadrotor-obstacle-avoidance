import os
import re
import sys
import copy
import time
import yaml
import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

if len(sys.argv) == 2:
    SEED = int(sys.argv[1])
else:
    SEED = 20220301
random.seed(SEED)
np.random.seed(SEED)

class Trajectory:
    def __init__(self, config, traj_type = "000000"):
        max_time = config['max_time']
        self.dt = config['dt']
        self.N = int(np.ceil(max_time/self.dt))

        if traj_type[-4:] == "0000" \
                and "r" not in traj_type \
                and "R" not in traj_type:
            self.N = 1
            self.is_static = True
        else:
            self.is_static = False

        self.name = None
        self.t = np.zeros((self.N ,1), dtype=float)
        self.t = np.arange(0, max_time, self.dt)

        self.config = config
        self.generateTrajectory(traj_type)
        self.integrate()
        self.boundingBox(config['pos_bb'])
        if isinstance(self.config['exclusion'][0], np.ndarray):
            [self.exclusionZone(zone) for zone in config['exclusion']]
        else:
            self.exclusionZone(config['exclusion'])


    def __hash__(self):
        return hash((self.t.data.tobytes(), self.pos.data.tobytes(), self.att.data.tobytes()))


    def eul2quat(self, rpy):
        roll = rpy[:,0]
        pitch = rpy[:,1]
        yaw = rpy[:,2]
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) \
                - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) \
                + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) \
                - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) \
                + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return np.array([qw, qx, qy, qz]).T


    def exclusionZone(self, corners):
        incursion = ((self.pos < corners[1,:]) & (self.pos > corners[0,:])).all(axis=-1)
        self.pos[incursion,:] = 1000


    def boundingBox(self, corners):
        outside = ((self.pos > corners[1,:]) | (self.pos < corners[0,:])).any(axis=-1)
        if (outside == True).any():
            self.i_max = max(1,np.argmax(outside))
        else:
            self.i_max = len(outside)


    def toCsv(self, basepath):
        if self.is_static:
            return
        assert(self.pos.shape[0] == self.att.shape[0]), "Attitude has wrong shape"
        assert(self.pos.shape[0] == self.N), "Time has wrong shape"

        if not os.path.exists(basepath):
            os.makedirs(basepath)

        fname = "traj_%s" % str(abs(hash(self)))
        path = os.path.join(basepath, fname)
        data = np.c_[self.t[:self.N], self.pos, self.att]
        data = data[:min(self.i_max, self.N),:]
        np.savetxt(path + ".csv", data, delimiter=",", header="header")
        self.name = fname


    def _parseIdentifier(self, identifier, bounds):
        dim = bounds.shape[1]
        if identifier == "0":
            arr = np.zeros((self.N,dim))
        elif identifier == "c" or identifier == "C":
            delta = bounds[1,:] - bounds[0,:]
            arr = np.tile(np.random.rand(1,dim) * delta + bounds[0,:], (self.N,1))
        elif identifier == "r" or identifier == "R":
            delta = bounds[1,:] - bounds[0,:]
            arr = np.random.rand(self.N,dim) * delta + bounds[0,:]
        else:
            print("ERR: Unable to parse identifier %s" % identifier)
            exit(-1)
        return arr


    def generateTrajectory(self, identifier):
        self.angacc = self._parseIdentifier(identifier[5], self.config['ang_bb'])
        self.acc = self._parseIdentifier(identifier[4], self.config['acc_bb'])

        self.omega = self._parseIdentifier(identifier[3], self.config['ome_bb'])
        self.vel = self._parseIdentifier(identifier[2], self.config['vel_bb'])

        self.eul = self._parseIdentifier(identifier[1], self.config['eul_bb'])
        self.pos = self._parseIdentifier(identifier[0], self.config['pos_bb'])


    def integrate(self):
        self.omega += np.cumsum(self.angacc.T, axis=-1).T * self.dt
        self.vel += np.cumsum(self.acc.T, axis=-1).T * self.dt
        self.eul += np.cumsum(self.omega.T, axis=-1).T * self.dt
        self.pos += np.cumsum(self.vel.T, axis=-1).T * self.dt
        self.att = self.eul2quat(self.eul * np.pi / 180)


    def plot(self):
        plt.figure()
        for i in range(3):
            plt.plot(self.t, self.pos[:,i])
        plt.legend(["X", "Y", "Z"])
        plt.show()


class Obstacle:
    def __init__(self, config):
        self.prefab = config['prefab_name']
        self.scale = np.random.uniform(low=config['scale'][0],
                                       high=config['scale'][1])
        self.traj_type = config['traj_type']
        self.trajectory = Trajectory(config, self.traj_type)


    def toDict(self):
        d = {}
        d['prefab'] = self.prefab
        d['position'] = [float(x) for x in list(self.trajectory.pos[0,:])]
        d['rotation'] = [float(x) for x in list(self.trajectory.att[0,:])]
        d['scale'] = 3*[self.scale]
        if self.trajectory.name is None:
            basepath = "csvtrajs"
            self.trajectory.toCsv(basepath)
        d['csvtraj'] = self.trajectory.name
        d['loop'] = False
        return d


    def isStatic(self):
        return self.trajectory.is_static


class ObstacleGroup:
    def __init__(self, global_config, local_config, name = "object"):
        """ The idea is that any global setting can be overwritten for any
        local group. So some things might only appear in certain places and
        not other places. Or mushrooms may have an exclusion zone or whatever.
        This is extremely flexibel and easy to expand.
        """
        self.obstacle_list = []
        assert("density" in local_config.keys()), "Density required for %s" % name
        if local_config['density'] == 0:
            return

        self.config = copy.deepcopy(global_config)

        assert("prefab_name" in local_config.keys()), "Prefab name required for %s" % name
        assert("traj_type" in local_config.keys()), "Trajectory type required for %s" % name
        for key in local_config.keys():
           self.config[key] = local_config[key]

        self.config['pos_bb'] = np.reshape(np.array(
                                 self.config['pos_bb'],dtype=float), (3,2)).T
        self.config['vel_bb'] = np.reshape(np.array(
                                 self.config['vel_bb'],dtype=float), (3,2)).T
        self.config['acc_bb'] = np.reshape(np.array(
                                 self.config['acc_bb'],dtype=float), (3,2)).T
        self.config['eul_bb'] = np.reshape(np.array(
                                 self.config['att_bb'],dtype=float), (3,2)).T
        self.config['ome_bb'] = np.reshape(np.array(
                                 self.config['omega_bb'],dtype=float), (3,2)).T
        self.config['ang_bb'] = np.reshape(np.array(
                                 self.config['angacc_bb'],dtype=float), (3,2)).T
        if isinstance(self.config['exclusion_zone'][0], list):
            self.config['exclusion'] = \
                    [np.reshape(np.array(zone), (3,2)).T for zone in config['exclusion_zone']]
        else:
            self.config['exclusion'] = \
                    np.reshape(np.array(config['exclusion_zone']), (3,2)).T

        self.bb = self.config['pos_bb']
        delta = self.bb[1,:] - self.bb[0,:]
        self.area = delta[0] * delta[1]
        self.volume = delta[0] * delta[1] * delta[2]
        self.num_objects = int(np.ceil(self.area * self.config['density']))
        for i in range(self.num_objects):
            self.obstacle_list.append(Obstacle(self.config))


    def getObstacleList(self):
        return self.obstacle_list


class World:
    def __init__(self, config):
        self.config = config

        self.obs_groups = []
        for key in self.config.keys():
            local_config = self.config[key]
            if isinstance(local_config, dict) and "prefab_name" in local_config.keys():
                self.obs_groups.append(ObstacleGroup(self.config, local_config, name=key))


    def toYaml(self, filename):
        d = {}
        obstacle_list = []
        for group in self.obs_groups:
            obstacle_list.extend(group.getObstacleList())

        i = 0
        static_obstacles = []
        for obstacle in obstacle_list:
            if obstacle.isStatic():
                o = obstacle.toDict()
                tmp = [o['prefab']]
                tmp.extend(o['position'])
                tmp.extend(o['rotation'])
                tmp.extend(o['scale'])
                static_obstacles.append(", ".join([str(x) for x in tmp]))
            else:
                i += 1
                d["Object%i" % i] = obstacle.toDict()

        d["N"] = len(d.keys())
        with open(filename, "w") as f:
            yaml.dump(d, f)

        with open("static_obstacles.csv", "w") as f:
            f.write("\n".join(static_obstacles))


class Plotter:
    def __init__(self, csvfolder, config):
        if not os.path.isdir(csvfolder):
            return

        csvs = [x for x in os.listdir(csvfolder) if os.path.splitext(x)[1] == ".csv"]

        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111, projection='3d')
        for csv in csvs:
            data = np.loadtxt(os.path.join(csvfolder, csv), delimiter = ",", skiprows = 1, ndmin=2)
            self.plotCsv(data)

        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
        self.ax.set_xlim((0, 60))
        self.ax.set_ylim((-25, 25))
        self.ax.set_zlim((0, 10))
        plt.show()


    def plotCsv(self, data):
        n = int(np.ceil(data.shape[0]/100))
        self.ax.scatter(data[::n,1], data[::n,2], data[::n,3])




if __name__=="__main__":
    os.system("rm -rf csvtrajs/*")
    with open("obstacle_config.yaml") as f:
        config = yaml.safe_load(f)
    w = World(config)
    w.toYaml("dynamic_obstacles.yaml")

    #  p = Plotter("csvtrajs", config)



