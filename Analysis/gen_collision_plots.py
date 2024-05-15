import numpy as np
import matplotlib.pyplot as plt

import glob, os, sys, time
from os.path import join as opj
from matplotlib import cm
from matplotlib.colors import LightSource

from mpl_toolkits.mplot3d.axes3d import Axes3D

from scipy.interpolate import make_interp_spline, BSpline

import seaborn as sns

class gen_plot_data(object):

    def __init__(self, traj_folders, ):

        self.num_folders = len(traj_folders)
        self.traj_metadata = []

        for folder in traj_folders:
            try:
                metadata = np.genfromtxt(opj(folder, "data.csv"), delimiter=",", dtype=np.float64)[1:, :-1]
                #coll_data = np.genfromtxt(opj(folder, "data.csv"), delimiter=",", dtype=np.float64)[1:, -1]
                coll_data = np.genfromtxt(opj(folder, "data.csv"), delimiter=",", dtype=bool)[1:, -1]
                coll_data = np.int32(coll_data)
                self.traj_metadata.append(np.column_stack((metadata, coll_data)))
            except:
                continue

        self.traj_metadata = np.row_stack(self.traj_metadata,)

        self.des_vel_sorted_data = {}

        #vels = [3,3.5,4.,4.5,5.,5.5,6.,6.5,7.]
        vels = [3.,4.,5.,6.,7.]

        for vel in vels:
            self.des_vel_sorted_data[vel] = self.traj_metadata[self.traj_metadata[:, 2] == vel, :]



    def get_collision_data(self,):

        stats_data = np.zeros((len(self.des_vel_sorted_data), 2))
        t_per_coll_data = np.zeros((len(self.des_vel_sorted_data), 2))

        for vel_ind, vel in enumerate(self.des_vel_sorted_data.keys()):
            
            traj_data = self.des_vel_sorted_data[vel]
            traj_start_indices = np.where(traj_data[:, 0] == 0)[0]

            num_collisions_in_traj = []
            timestamps_in_collision_list = []
            for start_indices in range(len(traj_start_indices)-1):
                
                individual_trajs_data = traj_data[traj_start_indices[start_indices]:traj_start_indices[start_indices+1], :]
                timestamps_in_collision = len(individual_trajs_data[individual_trajs_data[:, -1] == 1]) * 0.045
                
                obstacle_count = 0
                start_collision_t = 0.0
                for i in range(len(individual_trajs_data)-1):
                    if(individual_trajs_data[i, -1] == 1 and individual_trajs_data[i+1, -1] == 0):
                        # collision_time = self.traj_metadata[i, 1] - start_collision_t
                        # #obstacle_count += np.ceil(collision_time)
                        obstacle_count += 1
                        #total_obs_duration += collision_time
                
                num_collisions_in_traj.append(obstacle_count)
                timestamps_in_collision_list.append((timestamps_in_collision / obstacle_count) if obstacle_count else 0)

            stats_data[vel_ind, :] = np.array([np.mean(num_collisions_in_traj), np.std(num_collisions_in_traj)])
            t_per_coll_data[vel_ind] = np.array([np.mean(timestamps_in_collision_list), np.std(timestamps_in_collision_list)])

        return stats_data, t_per_coll_data

        # obstacle_count = 0.
        # total_obs_duration = 0.
        # total_obs_timesteps = 0

        # total_obs_timesteps = len(np.where(self.traj_metadata[:, -1] == True)[0])

        # start_collision_t = 0.0
        # for i in range(len(self.traj_metadata)-1):
        #     if(self.traj_metadata[i, -1] == False and self.traj_metadata[i+1, -1] == True):
        #         start_collision_t = self.traj_metadata[i, 1]
        #     if(self.traj_metadata[i, -1] == True and self.traj_metadata[i+1, -1] == False):
        #         collision_time = self.traj_metadata[i, 1] - start_collision_t
        #         #obstacle_count += np.ceil(collision_time)
        #         obstacle_count += 1
        #         total_obs_duration += collision_time
        #         # if(collision_time > 0.3): 
        #         #     obstacle_count += np.ceil(collision_time)

        # #print(f"Average collision count: {obstacle_count / 10}")

        # #print(traj_metadata)
                
        # ret_data1 = 0 if obstacle_count == 0 else obstacle_count / self.num_folders
        # ret_data2 = 0 if obstacle_count == 0 else total_obs_duration / (obstacle_count * self.num_folders)
        # ret_data3 = 0 if obstacle_count == 0 else total_obs_timesteps / (obstacle_count * self.num_folders)

        # return ret_data1, ret_data2, ret_data3, (obstacle_count, total_obs_duration, total_obs_timesteps, self.num_folders)
    
    def get_traj_stats(self,):
        
        x_bins = np.linspace(0, 59, 60)

        x_digitized = np.digitize(self.traj_metadata[:, 7], x_bins)

        traj_stats_data = np.zeros((len(x_bins), 4))

        for i in range(len(x_bins)):

            x_indices = np.where(x_digitized == i)[0]
            if not len(x_indices): continue
            corresponding_yz_data = self.traj_metadata[x_indices, 8:10]

            traj_stats_data[i, 0:2] = np.mean(corresponding_yz_data, axis=0)
            traj_stats_data[i, 2:] = np.std(corresponding_yz_data, axis=0)
            
        traj_stats_data = traj_stats_data[~np.all(traj_stats_data == 0, axis=1)]

        self.traj_stats = traj_stats_data
        
        return traj_stats_data, self.traj_metadata
    

if __name__ == "__main__":
    
    ####### MAKE SURE TO CHANGE PATHS APPROPRIATELY #####
    vit_folder = "./logs_trees/vit/"
    unet_folder = "./logs_trees/unet/"
    lstm_folder = "./logs_trees/lstm/"
    vitlstm_folder = "./logs_trees/vitlstm/"
    conv_folder = "./logs_trees/convnet/"


    ######## EXAMPLES FOR PATHS ##########
    # vit_folder = "./logs_modified_md/vit/"
    # unet_folder = "./logs_modified_md/unet/"
    # lstm_folder = "./logs_modified_md/lstm/"
    # vitlstm_folder = "./logs_modified_md/vitlstm/"
    # conv_folder = "./logs_modified_md/convnet/"
    #expert_folder = "./logs_modified_md/expert"

    # vit_folder = "./logs_no_md/vit/"
    # unet_folder = "./logs_no_md/unet/"
    # lstm_folder = "./logs_no_md/lstm/"
    # vitlstm_folder = "./logs_no_md/vitlstm/"
    # conv_folder = "./logs_no_md/convnet/"
    # expert_folder = "./logs_no_md/expert"

    # vit_folder = "./logs_nocol/vit/"
    # unet_folder = "./logs_nocol/unet/"
    # lstm_folder = "./logs_nocol/lstm/"
    # vitlstm_folder = "./logs_nocol/vitlstm/"
    # conv_folder = "./logs_nocol/convnet/"
    # expert_folder = "./logs_nocol/expert"

    vit_traj_folders = sorted(glob.glob(opj(vit_folder, "*")))
    unet_traj_folders = sorted(glob.glob(opj(unet_folder, "*")))
    lstm_traj_folders = sorted(glob.glob(opj(lstm_folder, "*")))
    vit_lstm_traj_folders = sorted(glob.glob(opj(vitlstm_folder, "*")))
    conv_traj_folders = sorted(glob.glob(opj(conv_folder, "*")))
    #expert_traj_folders = sorted(glob.glob(opj(expert_folder, "*")))

    vit_data = gen_plot_data(vit_traj_folders,)
    unet_data = gen_plot_data(unet_traj_folders,)
    lstm_data = gen_plot_data(lstm_traj_folders,)
    vitlstm_data = gen_plot_data(vit_lstm_traj_folders,)
    conv_data = gen_plot_data(conv_traj_folders,)
    #expert_data = gen_plot_data(expert_traj_folders,)

    vit_stats, vit_t_per_col = vit_data.get_collision_data()
    unet_stats, unet_t_per_col = unet_data.get_collision_data()
    lstm_stats, lstm_t_per_col = lstm_data.get_collision_data()
    vitlstm_stats, vitlstm_t_per_col = vitlstm_data.get_collision_data()
    conv_stats, conv_t_per_col = conv_data.get_collision_data()
    #expert_stats, expert_t_per_col = expert_data.get_collision_data()

    # According to the data available!
    # Make sure to change vels in the class too above
    #des_vels = np.array([3.,3.5,4.,4.5,5.,5.5,6.,6.5,7.])
    des_vels = np.array([3.,4.,5.,6.,7.])

    sns.set_context("talk")
    fig = plt.figure(num=1, figsize=(35/2, 25/2))

    plt.rc("xtick", labelsize=55)
    plt.rc("ytick", labelsize=55)
    plt.rc("axes", labelsize=55)
    plt.rc("axes", titlesize=60)
    plt.rc("legend", fontsize=42)
    
    ax = fig.gca()
    ax.grid(which = "major", linewidth = 1, alpha=1)
    ax.grid(which = "minor", linewidth = 0.2, alpha=0.2)
    ax.minorticks_on()

    plt.plot(des_vels, vitlstm_stats[:, 0], "g")
    plt.fill_between(des_vels, vitlstm_stats[:, 0] - 0.1 * vitlstm_stats[:, 1], vitlstm_stats[:, 0] + 0.1 * vitlstm_stats[:, 1], alpha=0.3, color="g", linestyle="dashed", label="_nolegend_")

    plt.plot(des_vels, lstm_stats[:, 0], "r")
    plt.fill_between(des_vels, lstm_stats[:, 0] - 0.1 * lstm_stats[:, 1], lstm_stats[:, 0] + 0.1 * lstm_stats[:, 1], alpha=0.3, color="r", linestyle="dashed", label="_nolegend_")

    plt.plot(des_vels, unet_stats[:, 0], "darkgoldenrod")
    plt.fill_between(des_vels, unet_stats[:, 0] - 0.1 * unet_stats[:, 1], unet_stats[:, 0] + 0.1 * unet_stats[:, 1], alpha=0.3, color="darkgoldenrod", linestyle="dashed", label="_nolegend_")

    plt.plot(des_vels, conv_stats[:, 0], "steelblue")
    plt.fill_between(des_vels, conv_stats[:, 0] - 0.1 * conv_stats[:, 1], conv_stats[:, 0] + 0.1 * conv_stats[:, 1], alpha=0.3, color="steelblue", linestyle="dashed", label="_nolegend_")

    plt.plot(des_vels, vit_stats[:, 0], "gray")
    plt.fill_between(des_vels, vit_stats[:, 0] - 0.1 * vit_stats[:, 1], vit_stats[:, 0] + 0.1 * vit_stats[:, 1], alpha=0.3, color="gray", linestyle="dashed", label="_nolegend_")

    #plt.plot(des_vels, expert_stats[:, 0], "saddlebrown")
    #plt.fill_between(des_vels, expert_stats[:, 0] - 0.1 * expert_stats[:, 1], expert_stats[:, 0] + 0.1 * expert_stats[:, 1], alpha=0.3, color="saddlebrown", linestyle="dashed", label="_nolegend_")


    plt.xlabel("Forward velocity (m/s)")
    plt.ylabel("Mean collision rate per trial")
    plt.title(r"Collision statistics $(\mu \pm 0.1\sigma)$")
    plt.legend(["ViT+LSTM", "LSTMnet", "Unet", "Convnet", "ViT", "Expert"], loc="best", fancybox=True)

    fig.tight_layout()

    # Example to save plots!
    plt.savefig("./plot_trees/trees_collision_plots.pdf",)
    plt.savefig("./plot_trees/trees_collision_plots.png", dpi=900)


    fig = plt.figure(2, figsize=(35/2, 25/2))
    sns.set_context("talk")

    plt.rc("xtick", labelsize=55)
    plt.rc("ytick", labelsize=55)
    plt.rc("axes", labelsize=55)
    plt.rc("axes", titlesize=60)
    plt.rc("legend", fontsize=42)

    ax = fig.gca()
    ax.grid(which = "major", linewidth = 1, alpha=1)
    ax.grid(which = "minor", linewidth = 0.2, alpha=0.2)
    ax.minorticks_on()

    plt.plot(des_vels, vitlstm_t_per_col[:, 0], "g")
    plt.fill_between(des_vels, vitlstm_t_per_col[:, 0] - 0.1 * vitlstm_t_per_col[:, 1], vitlstm_t_per_col[:, 0] + 0.1 * vitlstm_t_per_col[:, 1], alpha=0.3, color="g", linestyle="dashed", label="_nolegend_")

    plt.plot(des_vels, lstm_t_per_col[:, 0], "r")
    plt.fill_between(des_vels, lstm_t_per_col[:, 0] - 0.1 * lstm_t_per_col[:, 1], lstm_t_per_col[:, 0] + 0.1 * lstm_t_per_col[:, 1], alpha=0.3, color="r", linestyle="dashed", label="_nolegend_")

    plt.plot(des_vels, unet_t_per_col[:, 0], "darkgoldenrod")
    plt.fill_between(des_vels, unet_t_per_col[:, 0] - 0.1 * unet_t_per_col[:, 1], unet_t_per_col[:, 0] + 0.1 * unet_t_per_col[:, 1], alpha=0.3, color="darkgoldenrod", linestyle="dashed", label="_nolegend_")

    plt.plot(des_vels, conv_t_per_col[:, 0], "steelblue")
    plt.fill_between(des_vels, conv_t_per_col[:, 0] - 0.1 * conv_t_per_col[:, 1], conv_t_per_col[:, 0] + 0.1 * conv_t_per_col[:, 1], alpha=0.3, color="steelblue", linestyle="dashed", label="_nolegend_")

    plt.plot(des_vels, vit_t_per_col[:, 0], "gray")
    plt.fill_between(des_vels, vit_t_per_col[:, 0] - 0.1 * vit_t_per_col[:, 1], vit_t_per_col[:, 0] + 0.1 * vit_t_per_col[:, 1], alpha=0.3, color="gray", linestyle="dashed", label="_nolegend_")

    #plt.plot(des_vels, expert_t_per_col[:, 0], "saddlebrown")
    #plt.fill_between(des_vels, expert_t_per_col[:, 0] - 0.1 * expert_t_per_col[:, 1], expert_t_per_col[:, 0] + 0.1 * expert_t_per_col[:, 1], alpha=0.3, color="saddlebrown", linestyle="dashed", label="_nolegend_")


    plt.xlabel("Forward velocity (m/s)")
    plt.ylabel("Mean time (s) in collision")
    plt.title(r"Time in collision statistics $(\mu \pm 0.1\sigma)$")
    plt.legend(["ViT+LSTM", "LSTMnet", "Unet", "Convnet", "ViT", "Expert"], loc="upper right", fancybox=True)

    fig.tight_layout()

    plt.savefig("./plot_trees/trees_timesteps_per_collision_plots.pdf",)
    plt.savefig("./plot_trees/trees_timesteps_per_collision_plots.png", dpi=900)

    plt.show()


    ################# SMOOTH PLOTS (INTERPOLATE) #####################
    # des_vels = np.array([3.,3.5,4.,4.5,5.,5.5,6.,6.5,7.])
    # des_vels_smooth = np.linspace(des_vels.min(), des_vels.max(), 120)

    # vitlstm_mean_spl = make_interp_spline(des_vels, vitlstm_stats[:, 0], k=2)
    # vitlstm_mean_smooth = vitlstm_mean_spl(des_vels_smooth)

    # vitlstm_var_spl = make_interp_spline(des_vels, vitlstm_stats[:, 1], k=2)
    # vitlstm_var_smooth = vitlstm_var_spl(des_vels_smooth)

    # lstm_mean_spl = make_interp_spline(des_vels, lstm_stats[:, 0], k=2)
    # lstm_mean_smooth = lstm_mean_spl(des_vels_smooth)

    # lstm_var_spl = make_interp_spline(des_vels, lstm_stats[:, 1], k=2)
    # lstm_var_smooth = lstm_var_spl(des_vels_smooth)

    # unet_mean_spl = make_interp_spline(des_vels, unet_stats[:, 0], k=2)
    # unet_mean_smooth = unet_mean_spl(des_vels_smooth)

    # unet_var_spl = make_interp_spline(des_vels, unet_stats[:, 1], k=2)
    # unet_var_smooth = unet_var_spl(des_vels_smooth)

    # conv_mean_spl = make_interp_spline(des_vels, conv_stats[:, 0], k=2)
    # conv_mean_smooth = conv_mean_spl(des_vels_smooth)

    # conv_var_spl = make_interp_spline(des_vels, conv_stats[:, 1], k=2)
    # conv_var_smooth = conv_var_spl(des_vels_smooth)

    # vit_mean_spl = make_interp_spline(des_vels, vit_stats[:, 0], k=2)
    # vit_mean_smooth = vit_mean_spl(des_vels_smooth)

    # vit_var_spl = make_interp_spline(des_vels, vit_stats[:, 1], k=2)
    # vit_var_smooth = vit_var_spl(des_vels_smooth)

    # expert_mean_spl = make_interp_spline(des_vels, expert_stats[:, 0], k=2)
    # expert_mean_smooth = expert_mean_spl(des_vels_smooth)

    # expert_var_spl = make_interp_spline(des_vels, expert_stats[:, 1], k=2)
    # expert_var_smooth = expert_var_spl(des_vels_smooth)

    # fig, ax = plt.subplots()
    # ax.grid(which = "major", linewidth = 1, alpha=0.8)
    # ax.grid(which = "minor", linewidth = 0.2, alpha=0.9)
    # ax.minorticks_on()

    # plt.plot(des_vels_smooth, vitlstm_mean_smooth, "g")
    # plt.fill_between(des_vels_smooth, vitlstm_mean_smooth - 0.1 * vitlstm_var_smooth, vitlstm_mean_smooth + 0.1 * vitlstm_var_smooth, alpha=0.3, color="g", linestyle="dashed", label="_nolegend_")

    # plt.plot(des_vels_smooth, lstm_mean_smooth, "r")
    # plt.fill_between(des_vels_smooth, lstm_mean_smooth - 0.1 * lstm_var_smooth, lstm_mean_smooth + 0.1 * lstm_var_smooth, alpha=0.3, color="r", linestyle="dashed", label="_nolegend_")

    # plt.plot(des_vels_smooth, unet_mean_smooth, "darkgoldenrod")
    # plt.fill_between(des_vels_smooth, unet_mean_smooth - 0.1 * unet_var_smooth, unet_mean_smooth + 0.1 * unet_var_smooth, alpha=0.3, color="darkgoldenrod", linestyle="dashed", label="_nolegend_")

    # plt.plot(des_vels_smooth, conv_mean_smooth, "steelblue")
    # plt.fill_between(des_vels_smooth, conv_mean_smooth - 0.1 * conv_var_smooth, conv_mean_smooth + 0.1 * conv_var_smooth, alpha=0.3, color="steelblue", linestyle="dashed", label="_nolegend_")

    # plt.plot(des_vels_smooth, vit_mean_smooth, "gray")
    # plt.fill_between(des_vels_smooth, vit_mean_smooth - 0.1 * vit_var_smooth, vit_mean_smooth + 0.1 * vit_var_smooth, alpha=0.3, color="gray", linestyle="dashed", label="_nolegend_")

    # plt.plot(des_vels_smooth, expert_mean_smooth, "saddlebrown")
    # plt.fill_between(des_vels_smooth, vit_mean_smooth - 0.1 * vit_var_smooth, vit_mean_smooth + 0.1 * vit_var_smooth, alpha=0.3, color="saddlebrown", linestyle="dashed", label="_nolegend_")


    # plt.xlabel("Velocities (m/s)")
    # plt.ylabel("Average collision")
    # plt.title(r"Collision statistics $(\mu \pm 0.1\sigma)$")
    # plt.legend(["ViT + LSTM (3L)", "LSTMNet (5L)", "UNet + LSTM (2L)", "ConvNet", "ViT",])

    # fig.tight_layout()

    # plt.savefig("./plots/collision_plots_sm.eps", format="eps")
    # plt.savefig("./plots/collision_plots_sm.png", dpi=900)

    # #plt.grid(True)
    # plt.show()