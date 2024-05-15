import numpy as np
import matplotlib.pyplot as plt

import glob, os, sys, time
from os.path import join as opj
from matplotlib import cm
from matplotlib.colors import LightSource

import seaborn as sns

from mpl_toolkits.mplot3d.axes3d import Axes3D

class gen_plot_data(object):

    def __init__(self, traj_folders, obs_folder):

        self.num_folders = len(traj_folders)
        self.traj_metadata = []

        for folder in traj_folders:
            try:
                metadata = np.genfromtxt(opj(folder, "data.csv"), delimiter=",", dtype=np.float64)[1:, :-1]
                coll_data = np.genfromtxt(opj(folder, "data.csv"), delimiter=",", dtype=bool)[1:, -1]
                self.traj_metadata.append(np.column_stack((metadata, coll_data)))
            except:
                continue

        self.traj_metadata = np.row_stack(self.traj_metadata, dtype=np.float64)

        self.obs_trajdata = np.genfromtxt(opj(obs_folder, "static_obstacles.csv"), delimiter=",", dtype=np.float64)[:, 1:]

    def get_collision_count(self,):

        obstacle_count = 0.
        total_obs_duration = 0.
        total_obs_timesteps = 0

        total_obs_timesteps = len(np.where(self.traj_metadata[:, -1] == True)[0])

        start_collision_t = 0.0
        for i in range(len(self.traj_metadata)-1):
            if(self.traj_metadata[i, -1] == False and self.traj_metadata[i+1, -1] == True):
                start_collision_t = self.traj_metadata[i, 1]
            if(self.traj_metadata[i, -1] == True and self.traj_metadata[i+1, -1] == False):
                collision_time = self.traj_metadata[i, 1] - start_collision_t
                #obstacle_count += np.ceil(collision_time)
                obstacle_count += 1
                total_obs_duration += collision_time
                # if(collision_time > 0.3): 
                #     obstacle_count += np.ceil(collision_time)

        #print(f"Average collision count: {obstacle_count / 10}")

        #print(traj_metadata)
                
        ret_data1 = 0 if obstacle_count == 0 else obstacle_count / self.num_folders
        ret_data2 = 0 if obstacle_count == 0 else total_obs_duration / (obstacle_count * self.num_folders)
        ret_data3 = 0 if obstacle_count == 0 else total_obs_timesteps / (obstacle_count * self.num_folders)

        return ret_data1, ret_data2, ret_data3, (obstacle_count, total_obs_duration, total_obs_timesteps, self.num_folders)
    
    
    def get_traj_stats(self,):
        
        x_bins = np.linspace(0, 59, 60)

        x_digitized = np.digitize(self.traj_metadata[:, 7], x_bins)

        traj_stats_data = np.zeros((len(x_bins), 4))

        for i in range(len(x_bins)):

            x_indices = np.where(x_digitized == i)[0]
            if not len(x_indices): continue
            corresponding_yz_data = self.traj_metadata[x_indices, 8:10]

            #print(x_indices)
            #print(corresponding_yz_data)

            traj_stats_data[i, 0:2] = np.mean(corresponding_yz_data, axis=0)
            traj_stats_data[i, 2:] = np.std(corresponding_yz_data, axis=0)

        #print(x_digitized)
            
        traj_stats_data = traj_stats_data[~np.all(traj_stats_data == 0, axis=1)]

        self.traj_stats = traj_stats_data
        
        return traj_stats_data, self.traj_metadata

    def plot_sphere(self, pos, radius, yz_mean, proj=True, alpha=0.5):
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x = radius * np.cos(u)*np.sin(v) + pos[0]
        # y = (radius/5) * np.sin(u)*np.sin(v) + pos[1]
        # z = (radius/10) * np.cos(v) + pos[2]
        y = (radius/5) * np.sin(u)*np.sin(v) + pos[1]
        z = (radius/10) * np.cos(v) + pos[2]

        #rgb = ls.shade(x, cmap=cm.Wistia, vert_exag=0.1, blend_mode='soft')
        # blend shade
        #bsl = ls.blend_hsv(rgb, np.expand_dims(x*0.8, 2))

        ax.plot_surface(x, y, z, color='k', alpha=alpha,)
        if proj:
            ax.contourf(x, y, z, zdir="z", offset=0, colors='r', alpha=0.3)
            ax.contourf(x, y, z, zdir="y", offset=5, colors='r', alpha=0.3)

    def plot_2d3d_traj(self ,c1="g", c2="k", with_obs=True):

        self.get_traj_stats()

        for i in range(len(self.traj_stats)-1):

            u = np.linspace(i*1, (i+1)*1, 20)
            t = np.linspace(0, 2 * np.pi, 20)

            t_c, u_c = np.meshgrid(t, u)

            std_dev_y_line = self.traj_stats[i, 2] + (( self.traj_stats[i+1, 2] - self.traj_stats[i, 2] ) / (1)) * (u_c - (i * 1))
            std_dev_z_line = self.traj_stats[i, 3] + (( self.traj_stats[i+1, 3] - self.traj_stats[i, 3] ) / (1)) * (u_c - (i * 1))

            mean_y_line = self.traj_stats[i, 0] + (( self.traj_stats[i+1, 0] - self.traj_stats[i, 0] ) / (1)) * (u_c - (i * 1))
            mean_z_line = self.traj_stats[i, 1] + (( self.traj_stats[i+1, 1] - self.traj_stats[i, 1] ) / (1)) * (u_c - (i * 1))

            #x_c = u_c * std_dev_y_line * (np.cos(t_c))  + mean_y_line
            #y_c = u_c * std_dev_z_line * (np.sin(t_c))  + mean_z_line
            x_c = std_dev_y_line * (np.cos(t_c))  + mean_y_line
            y_c = std_dev_z_line * (np.sin(t_c))  + mean_z_line

            ax.plot_surface(u_c, x_c, y_c, alpha=0.3, color=c1,)
            ax.contourf(u_c, x_c, y_c, zdir="z", offset=0, colors=c1, alpha=0.3)
            ax.contourf(u_c, x_c, y_c, zdir="y", offset=5, colors=c1, alpha=0.3)

            if with_obs and i%4 == 0:
                curr_xyz = np.insert(self.traj_stats[i, 0:2], 0, i)
                #Get the indices of two closest obstacles to the curr_xyz
                closest_obs_indices = np.argpartition(np.linalg.norm(curr_xyz - self.obs_trajdata[:, 0:3], axis=1), 1)[:2]

                for k in range(len(closest_obs_indices)):
                    self.plot_sphere(self.obs_trajdata[closest_obs_indices[k], 0:3], self.obs_trajdata[closest_obs_indices[k], -1], self.traj_stats[i, 0:2])
        
        ax.plot(range(len(self.traj_stats)), self.traj_stats[:, 0], self.traj_stats[:, 1], color=c2)
    
    def plot_2d_xy_traj(self, color="r"):
        self.get_traj_stats()
        x_axis = np.linspace(0, 59, self.traj_stats.shape[0])

        ax.plot(x_axis, self.traj_stats[:, 0], color=color)
        ax.fill_between(x_axis, self.traj_stats[:, 0] - self.traj_stats[:, 2], self.traj_stats[:, 0] + self.traj_stats[:, 2], color=color, alpha=0.3, linestyle="dashed", label="_nolegend_")

    def plot_2d_xz_traj(self, color="r"):
        self.get_traj_stats()
        x_axis = np.linspace(0, 59, self.traj_stats.shape[0])

        ax.plot(x_axis, self.traj_stats[:, 1], color=color)
        ax.fill_between(x_axis, self.traj_stats[:, 1] - self.traj_stats[:, 3], self.traj_stats[:, 1] + self.traj_stats[:, 3], color=color, alpha=0.3, linestyle="dashed", label="_nolegend_")

    def plot_3d_traj(self, color="r", with_obs=True):
        self.get_traj_stats()
        x_axis = np.linspace(0, 59, self.traj_stats.shape[0])
        y_axis = self.traj_stats[:, 0]
        z_axis = self.traj_stats[:, 1]

        ax.plot(x_axis, y_axis, z_axis, color=color)

        for i in range(len(self.traj_stats)-1):
            if with_obs and i%5 == 0:
                    curr_xyz = np.insert(self.traj_stats[i, 0:2], 0, i)
                    #Get the indices of two closest obstacles to the curr_xyz
                    closest_obs_indices = np.argpartition(np.linalg.norm(curr_xyz - self.obs_trajdata[:, 0:3], axis=1), 1)[:2]

                    for k in range(len(closest_obs_indices)):
                        self.plot_sphere(self.obs_trajdata[closest_obs_indices[k], 0:3], self.obs_trajdata[closest_obs_indices[k], -1], self.traj_stats[i, 0:2], proj=False, alpha=0.2)

        


if __name__ == "__main__":
    
    vit_folder = "./data/vit/"
    expert_folder = "./data/expert/"
    lstm_folder = "./data/lstm_modified_md/"
    vit_lstm_folder = "./data/vitlstm/"
    unet_folder = "./data/unet_modified_md/"
    conv_folder = "./data/convnet/"

    obstacle_folder = "/home/jarvis/Downloads/Softwares/build/catkin_ws/src/agile_flight/flightmare/flightpy/configs/vision/medium/environment_30/"

    vit_traj_folders = sorted(glob.glob(opj(vit_folder, "*")))
    expert_traj_folders = sorted(glob.glob(opj(expert_folder, "*")))
    lstm_traj_folders = sorted(glob.glob(opj(lstm_folder, "*")))
    vit_lstm_traj_folders = sorted(glob.glob(opj(vit_lstm_folder, "*")))
    conv_traj_folders = sorted(glob.glob(opj(conv_folder, "*")))
    unet_traj_folders = sorted(glob.glob(opj(unet_folder, "*")))

    vit_data = gen_plot_data(vit_traj_folders, obstacle_folder)
    expert_data = gen_plot_data(expert_traj_folders, obstacle_folder)
    lstm_data = gen_plot_data(lstm_traj_folders, obstacle_folder)
    vitlstm_data = gen_plot_data(vit_lstm_traj_folders, obstacle_folder)
    conv_data = gen_plot_data(conv_traj_folders, obstacle_folder)
    unet_data = gen_plot_data(unet_traj_folders, obstacle_folder)

    #ele, azim = 22, -49
    ele, azim = 32, -48

    #### PLOT ALL DATA (2D projections and 3D) IN ONE PLOT ######
    #### UNCOMMENT IF YOU WANT TO USE IT! ######################

    #plt.rcParams.update({'font.size': 50, 'font.family': 'serif'})

    # plt.rc('xtick', labelsize=50)
    # plt.rc('ytick', labelsize=50)
    # plt.rc('axes', labelsize=50)
    # plt.rc('axes', titlesize=150)
    # plt.rc('legend', fontsize=100)

    # fig = plt.figure(2)
    # fig.tight_layout()
    # ax = plt.axes(projection='3d')
    # #ax.view_init(elev=22, azim=-49)
    # ax.view_init(elev=ele, azim=azim)

    # vit_data.plot_2d3d_traj()
    
    # ax.set_xlim([0, 60])
    # ax.set_ylim([-5, 5])
    # ax.set_zlim([0, 5])
    # #ax.set_aspect('auto', adjustable='box')
    # #ax.axis("equal")
    # #ax.set_box_aspect([1.0, 1.0, 1.0])
    # ax.set_xlabel("x-axis (m)")
    # ax.set_ylabel("y-axis (m)")
    # ax.set_zlabel("z-axis (m)")
    # ax.set_title("ViT Trajectory Statistics")
    # plt.savefig("./plots_modified_md/vit/traj_2d_3d.pdf")
    # plt.savefig("./plots_modified_md/vit/traj_2d_3d.png", dpi=900)

    # fig = plt.figure(3)
    # fig.tight_layout()
    # ax = plt.axes(projection='3d')
    # ax.view_init(elev=ele, azim=azim)
    # expert_data.plot_2d3d_traj()
    # ax.set_xlim([0, 60])
    # ax.set_ylim([-5, 5])
    # ax.set_zlim([0, 5])
    # ax.set_xlabel("x-axis (m)")
    # ax.set_ylabel("y-axis (m)")
    # ax.set_zlabel("z-axis (m)")
    # ax.set_title("Expert Trajectory Statistics")
    # plt.savefig("./plots_modified_md/expert/traj_2d_3d.pdf")
    # plt.savefig("./plots_modified_md/expert/traj_2d_3d.png", dpi=900)

    # fig = plt.figure(4)
    # fig.tight_layout()
    # ax = plt.axes(projection='3d')
    # ax.view_init(elev=ele, azim=azim)
    # lstm_data.plot_2d3d_traj()
    # ax.set_xlim([0, 60])
    # ax.set_ylim([-5, 5])
    # ax.set_zlim([0, 5])
    # ax.set_xlabel("x-axis (m)")
    # ax.set_ylabel("y-axis (m)")
    # ax.set_zlabel("z-axis (m)")
    # ax.set_title("LSTMNet (5L) Trajectory Statistics")
    # plt.savefig("./plots_modified_md/lstm/traj_2d_3d.pdf")
    # plt.savefig("./plots_modified_md/lstm/traj_2d_3d.png", dpi=900)

    # fig = plt.figure(5)
    # fig.tight_layout()
    # ax = plt.axes(projection='3d')
    # ax.view_init(elev=ele, azim=azim)
    # vitlstm_data.plot_2d3d_traj()
    # ax.set_xlim([0, 60])
    # ax.set_ylim([-5, 5])
    # ax.set_zlim([0, 5])
    # ax.set_xlabel("x-axis (m)")
    # ax.set_ylabel("y-axis (m)")
    # ax.set_zlabel("z-axis (m)")
    # ax.set_title("ViT + LSTM (3L) Trajectory Statistics")
    # plt.savefig("./plots_modified_md/vitlstm/traj_2d_3d.pdf")
    # plt.savefig("./plots_modified_md/vitlstm/traj_2d_3d.png", dpi=900)

    # fig = plt.figure(6)
    # fig.tight_layout()
    # ax = plt.axes(projection='3d')
    # ax.view_init(elev=ele, azim=azim)
    # unet_data.plot_2d3d_traj()
    # ax.set_xlim([0, 60])
    # ax.set_ylim([-5, 5])
    # ax.set_zlim([0, 5])
    # ax.set_xlabel("x-axis (m)")
    # ax.set_ylabel("y-axis (m)")
    # ax.set_zlabel("z-axis (m)")
    # ax.set_title("UNet + LSTM (2L) Trajectory Statistics")
    # plt.savefig("./plots_modified_md/unet/traj_2d_3d.pdf")
    # plt.savefig("./plots_modified_md/unet/traj_2d_3d.png", dpi=900)

    # fig = plt.figure(7)
    # fig.tight_layout()
    # ax = plt.axes(projection='3d')
    # ax.view_init(elev=ele, azim=azim)
    # conv_data.plot_2d3d_traj()
    # ax.set_xlim([0, 60])
    # ax.set_ylim([-5, 5])
    # ax.set_zlim([0, 5])
    # ax.set_xlabel("x-axis (m)")
    # ax.set_ylabel("y-axis (m)")
    # ax.set_zlabel("z-axis (m)")
    # ax.set_title("ConvNet Trajectory Statistics")
    # plt.savefig("./plots_modified_md/convnet/traj_2d_3d.pdf")
    # plt.savefig("./plots_modified_md/convnet/traj_2d_3d.png", dpi=900)


    #fig = plt.figure(8)

    fig = plt.figure(num=8, figsize=(35/2, 25/2))

    sns.set_context("talk")

    plt.rc('xtick', labelsize=50)
    plt.rc('ytick', labelsize=50)
    plt.rc('axes', labelsize=50)
    plt.rc('axes', titlesize=55)
    plt.rc('legend', fontsize=35)

    fig.tight_layout()
    ax = fig.gca()
    ax.grid(which = "major", linewidth = 1, alpha=1.)
    ax.grid(which = "minor", linewidth = 0.2, alpha=0.2)
    ax.minorticks_on()
    
    vitlstm_data.plot_2d_xy_traj(color="g")
    lstm_data.plot_2d_xy_traj(color="r")
    vit_data.plot_2d_xy_traj(color="gray")
    expert_data.plot_2d_xy_traj(color="saddlebrown")
    unet_data.plot_2d_xy_traj(color="darkgoldenrod")
    conv_data.plot_2d_xy_traj(color="steelblue")

    ax.set_xlabel("x-axis (m)", labelpad=-3.0)
    ax.set_ylabel("y-axis (m)")
    ax.set_title("Variation of trajectory on the x-y plane")
    plt.legend(["ViT+LSTM", "LSTMnet", "ViT", "Expert", "Unet", "Convnet"], loc="best", fancybox=True)
    plt.savefig("./plots_modified_md/traj_2d_xy.pdf")
    plt.savefig("./plots_modified_md/traj_2d_xy.png", dpi=900)


    #fig = plt.figure(9)
    fig = plt.figure(num=9, figsize=(35/2, 25/2))

    sns.set_context("talk")

    plt.rc('xtick', labelsize=50)
    plt.rc('ytick', labelsize=50)
    plt.rc('axes', labelsize=50)
    plt.rc('axes', titlesize=55)
    plt.rc('legend', fontsize=35)

    fig.tight_layout()
    ax = fig.gca()
    ax.grid(which = "major", linewidth = 1, alpha=1)
    ax.grid(which = "minor", linewidth = 0.2, alpha=0.2)
    ax.minorticks_on()

    vitlstm_data.plot_2d_xz_traj(color="g")
    lstm_data.plot_2d_xz_traj(color="r")
    vit_data.plot_2d_xz_traj(color="gray")
    expert_data.plot_2d_xz_traj(color="saddlebrown")
    unet_data.plot_2d_xz_traj(color="darkgoldenrod")
    conv_data.plot_2d_xz_traj(color="steelblue")

    ax.set_xlabel("x-axis (m)", labelpad=-3.0)
    ax.set_ylabel("z-axis (m)")
    ax.set_title("Variation of trajectory on the x-z plane")
    plt.legend(["ViT+LSTM", "LSTMnet", "ViT", "Expert", "Unet", "Convnet"], loc="best", fancybox=True)
    plt.savefig("./plots_modified_md/traj_2d_xz.pdf")
    plt.savefig("./plots_modified_md/traj_2d_xz.png", dpi=900)


    #fig = plt.figure(10)
    fig = plt.figure(num=10, figsize=(35/2, 25/2))

    sns.set_context("talk")

    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('axes', labelsize=25)
    plt.rc('axes', titlesize=35)
    plt.rc('legend', fontsize=24)

    fig.tight_layout()
    ax = plt.axes(projection='3d')
    ax.view_init(elev=ele, azim=azim)

    vitlstm_data.plot_3d_traj(color="g")
    lstm_data.plot_3d_traj(color="r")
    vit_data.plot_3d_traj(color="gray")
    expert_data.plot_3d_traj(color="saddlebrown")
    unet_data.plot_3d_traj(color="darkgoldenrod")
    conv_data.plot_3d_traj(color="steelblue")

    ax.set_xlim([0, 60])
    ax.set_ylim([-5, 5])
    ax.set_zlim([0, 5])
    ax.set_xlabel("x-axis (m)", labelpad=10.0)
    ax.set_ylabel("y-axis (m)", labelpad=10.0)
    ax.set_zlabel("z-axis (m)", labelpad=10.0)
    ax.set_title("Trajectories with obstacles")
    plt.legend(["ViT+LSTM", "LSTMnet", "ViT", "Expert", "Unet", "Convnet"], loc="best", fancybox=True)
    plt.savefig("./plots_modified_md/traj_3d.pdf")
    plt.savefig("./plots_modified_md/traj_3d.png", dpi=900)


    #traj_start_indices = np.where(traj_data[:, 0] == 0)[0]

    # # Plot Trajectories
    # for i in range(len(traj_start_indices)-1):
    #     ax.plot3D(traj_data[traj_start_indices[i]:traj_start_indices[i+1], 7], traj_data[traj_start_indices[i]:traj_start_indices[i+1], 8], traj_data[traj_start_indices[i]:traj_start_indices[i+1], 9], )

    #plt.show()