import numpy as np
import matplotlib.pyplot as plt
import glob, os, sys, time
from os.path import join as opj
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# plt.rcParams.update({'font.size': 40, 'font.family': 'serif'})
# plt.rc('xtick', labelsize=40)
# plt.rc('ytick', labelsize=400)
# plt.rc('legend', fontsize=28)

class metrics:
    def __init__(self, folders, desiredVel):
        self.trajFolders = folders
        self.dataCSV = [opj(i, 'data.csv') for i in self.trajFolders]
        self.desiredVels = desiredVel       
 
    
    def successRate(self):
        output = [0] * len(self.desiredVels)
        numRuns = [0] * len(self.desiredVels)
        #print(numRuns)
        for file in self.dataCSV:
            data = pd.read_csv(file)

            if data['desired_vel'][0] not in self.desiredVels:
                continue
            
            NumCollisions = data['is_collide'].sum()
            desiredVel = data['desired_vel'][0]
            output[self.desiredVels.index(desiredVel)] += min(NumCollisions,1)
            if data['pos_x'][len(data)-1] < 30:

                continue
            numRuns[self.desiredVels.index(desiredVel)] += 1
        output = [output[i]/numRuns[i] for i in range(len(output))]
        output = [(1-i)*100 for i in output]
        return output

    def cal_energy_cost(self, dis, data):
        ts = data['timestamp'].to_numpy()
        dt = np.diff(ts)  # time differences
        vels = data[['vel_y', 'vel_z']].to_numpy()
        
        energy = 0
        for i in range(len(vels) - 2):
            if dt[i] <= 1e-5:  # Skip if time difference is too small
                continue
            
            ai = (vels[i + 1] - vels[i]) / dt[i]
            
            # Find next valid time difference
            j = i + 1
            while j < len(vels) - 2 and dt[j] <= 1e-5:
                j += 1
            if j >= len(vels) - 2:
                break
            
            aj = (vels[j + 1] - vels[j]) / dt[j]
            
            jerk = aj - ai
            energy += np.linalg.norm(jerk)
        return energy 
    
    def jerkCost(self):
        output = [0] * len(self.desiredVels)
        numRuns = [0] * len(self.desiredVels)
        for file in self.dataCSV:
            data = pd.read_csv(file)
            if data['desired_vel'][0] not in self.desiredVels:
                continue
            if data['pos_x'][len(data)-1] < 30: #### We ignore the folder where we have bridge failed !! ####
                continue

            ts = data['timestamp']
            dt = np.diff(ts)[:, None]
            vels = data[['vel_x', 'vel_y', 'vel_z']].to_numpy()
            # vels = np.apply_along_axis(lambda m: np.convolve(m, np.ones(5)/5, mode='same'), axis=0, arr=vels)
            acc = (np.diff(vels, axis = 0)*dt)/(dt*dt + 1e-9)            
            jerk = np.linalg.norm(np.diff(acc, axis=0), axis=1).sum() #Dont need dt because we integrate it anyway

            pos = data[['pos_x', 'pos_y', 'pos_z']].to_numpy()[data['pos_x']>0.1]
            dpos = np.diff(pos, axis = 0)
            distanceTravelled = np.linalg.norm(dpos, axis=1).sum()      

            desiredVel = data['desired_vel'][0]
            output[self.desiredVels.index(desiredVel)] += jerk/distanceTravelled
            numRuns[self.desiredVels.index(desiredVel)] += 1
        output = [output[i]/(numRuns[i]) for i in range(len(output))]
        return output

    def commandedAccel(self):
        output = [0] * len(self.desiredVels)
        numRuns = [0] * len(self.desiredVels)
        for file in self.dataCSV:
            data = pd.read_csv(file)
            if data['desired_vel'][0] not in self.desiredVels:
                continue

            if data['desired_vel'][0] > 7:
                continue
            if data['pos_x'][len(data)-1] < 30:
                continue
            ts = data['timestamp'][data['pos_x']>0.1]
            vels = data[['velcmd_x', 'velcmd_y', 'velcmd_z']].to_numpy()[data['pos_x']>0.1]
            vels = np.apply_along_axis(lambda m: np.convolve(m, np.ones(4)/4, mode='same'), axis=0, arr=vels)

            dt = np.diff(ts)[:, None]
            acc = np.diff(vels, axis = 0)*dt/(dt*dt + 1e-9)             
            normAccel = np.linalg.norm(acc, axis=1).sum() 

            pos = data[['pos_x', 'pos_y', 'pos_z']].to_numpy()[data['pos_x']>0.1]
            dpos = np.diff(pos, axis = 0)
            distanceTravelled = np.linalg.norm(dpos, axis=1).sum()      

            desiredVel = data['desired_vel'][0]
            output[self.desiredVels.index(desiredVel)] += normAccel/distanceTravelled
            numRuns[self.desiredVels.index(desiredVel)] += 1
        output = [output[i]/(numRuns[i]) for i in range(len(output))]
        return output
    
    def totalDistance(self):
        output = [0] * len(self.desiredVels)
        numRuns = [0] * len(self.desiredVels)
        for file in self.dataCSV:
            data = pd.read_csv(file)
            if data['pos_x'][len(data)-1] < 30:
                continue
            pos = data[['pos_x', 'pos_y', 'pos_z']].to_numpy()
            vel = np.diff(pos, axis = 0)
            pos = np.linalg.norm(vel, axis=1).sum()
            desiredVel = data['desired_vel'][0]
            output[self.desiredVels.index(desiredVel)] += pos
            numRuns[self.desiredVels.index(desiredVel)] += 1
        output = [output[i]/(numRuns[i]+1e-7) for i in range(len(output))]
        return output
    
    def timeTaken(self):
        output = [0] * len(self.desiredVels)
        numRuns = [0] * len(self.desiredVels)
        for file in self.dataCSV:
            data = pd.read_csv(file)
            if data['pos_x'][len(data)-1] < 30:
                print("expert!?")
                continue
            ts = data['timestamp'].to_numpy()[data['pos_x']>0.5]
            desiredVel = data['desired_vel'][0]
            output[self.desiredVels.index(desiredVel)] += (ts[-1] - ts[0]) + 60*(min(data['is_collide'].sum(),1))
            numRuns[self.desiredVels.index(desiredVel)] += 1
        output = [output[i]/numRuns[i] for i in range(len(output))]
        return output



def plotIt_sr(success_rates, velocities, whatToPlot):

    sns.set_context('talk')
    model_names = success_rates.keys()
    palette = sns.color_palette("tab10", len(model_names))
    #size = (35/3.75, 25/3.75)
    size = (35/2, 25/2)
    plt.figure(figsize=size)
    
    # plt.rc("xtick", labelsize=52)
    # plt.rc("ytick", labelsize=52)
    # plt.rc("axes", labelsize=52)
    # plt.rc("axes", titlesize=57)
    # plt.rc("legend", fontsize=40)

    plt.rc("xtick", labelsize=55)
    plt.rc("ytick", labelsize=55)
    plt.rc("axes", labelsize=55)
    plt.rc("axes", titlesize=60)
    plt.rc("legend", fontsize=42)

    ax = plt.gca()
    ax.grid(which="major", linestyle='-', linewidth=1, alpha=1)
    ax.grid(which="minor", linestyle=':', linewidth=0.4, alpha=0.5)
    ax.minorticks_on()

    markerStyle = ['o', 'v', 's', '*', 'h', 'd']
    for model, color, markers in zip(model_names, palette, markerStyle):
        
        # if 'Expert' in model:
        #     plt.plot([3,3.5,4,4.5,5,5.5,6,6.5,7], success_rates[model], label=model, color=color, linewidth=3.0, marker = markers, markerfacecolor='none')
        #     continue
        # else:
        plt.plot(velocities, success_rates[model], label=model, color=color, linewidth=3.0, marker = markers, markerfacecolor='none')

    plt.legend(loc='best', fancybox=True, ncol=3, columnspacing=0.6)
    plt.xlabel('Forward velocity (m/s)')
    plt.ylabel(r'Success rates $(\%)$')
    plt.title('Success rates per velocities')
    plt.xlim([3,7])
    plt.ylim([-2,102])

    # for label in ax.get_xticklabels(which='both'):
    #     label.set_fontsize(25)
    # for label in ax.get_yticklabels(which='both'):
    #     label.set_fontsize(25)

    plt.tight_layout()
    
    plt.savefig('trees_metadata_sr.png', dpi=300, bbox_inches='tight')
    plt.savefig('trees_metadata_sr.pdf', bbox_inches='tight')

    # plt.savefig('spheres_metadata_sr.png', dpi=300, bbox_inches='tight')
    # plt.savefig('spheres_metadata_sr.pdf', bbox_inches='tight')
    



def plotIt_energy(success_rates, velocities, whatToPlot):

    sns.set_context('talk')
    model_names = success_rates.keys()
    palette = sns.color_palette("tab10", len(model_names))
    plt.figure(figsize=(35/3.75, 25/3.75))
    
    plt.rc("xtick", labelsize=30)
    plt.rc("ytick", labelsize=30)
    plt.rc("axes", labelsize=30)
    plt.rc("axes", titlesize=40)
    plt.rc("legend", fontsize=18)

    markerStyle = ['o', 'v', 's', '*', 'h', 'd']
    for model, color, markers in zip(model_names, palette, markerStyle):
        plt.plot(velocities, success_rates[model], label=model, color=color, linewidth=3.0, marker = markers, markerfacecolor='none')
    ax = plt.gca()
    ax.grid(which="major", linestyle='-', linewidth=1, alpha=1)
    ax.grid(which="minor", linestyle=':', linewidth=0.4, alpha=0.5)
    ax.minorticks_on()

  

    plt.legend(loc='best', fancybox=True, ncol=3)
    plt.xlabel('Forward velocity (m/s)')
    plt.ylabel(r'Energy cost $(m/s^2)$')
    plt.title('Energy cost \n across velocities')
    plt.xlim([3,7])
    plt.ylim([3,12.5])

    # for label in ax.get_xticklabels(which='both'):
    #     label.set_fontsize(25)
    # for label in ax.get_yticklabels(which='both'):
    #     label.set_fontsize(25)

    plt.tight_layout()
    
    # plt.savefig('success_rates_across_velocities.pdf', dpi=300, bbox_inches='tight')
    # #plt.savefig(f"{whatToPlot}.eps", format="eps")

    plt.savefig('spheres_metadata_energy.png', dpi=900, bbox_inches='tight')
    plt.savefig('spheres_metadata_energy.pdf', bbox_inches='tight')

def plotIt(success_rates, velocities, whatToPlot):
    sns.set_context('talk')
    model_names = success_rates.keys()
    palette = sns.color_palette("tab10", len(model_names))
    size = (35/3.75, 25/3.75)
    plt.figure(figsize=size)
    
    plt.rc("xtick", labelsize=30)
    plt.rc("ytick", labelsize=30)
    plt.rc("axes", labelsize=30)
    plt.rc("axes", titlesize=40)
    plt.rc("legend", fontsize=18)

    markerStyle = ['o', 'v', 's', '*', 'h', 'd']
    for model, color, markers in zip(model_names, palette, markerStyle):
        plt.plot(velocities, success_rates[model], label=model, color=color, linewidth=3.0, marker = markers, markerfacecolor='none')
    ax = plt.gca()
    ax.grid(which="major", linestyle='-', linewidth=1, alpha=1)
    ax.grid(which="minor", linestyle=':', linewidth=0.4, alpha=0.5)
    ax.minorticks_on()

    plt.legend(loc='best', fancybox=True, ncol=3,)
    plt.xlabel('Forward velocity (m/s)',)
    plt.ylabel(r'Acceleration $(m/s^2)$',)
    plt.title('Commanded accelerations \n across velocities',)
    plt.xlim([3,7])
    plt.ylim([7,40])

    # for label in ax.get_xticklabels(which='both'):
    #     label.set_fontsize(25)
    # for label in ax.get_yticklabels(which='both'):
    #     label.set_fontsize(25)

    plt.tight_layout()
    
    plt.savefig('success_rates_across_velocities.png', dpi=900, bbox_inches='tight')
    plt.savefig('success_rates_across_velocities.pdf', bbox_inches='tight')


def main():

    #Trees
    vit_folder = "./logs/vit/"
    unet_folder = "./logs/unet/"
    lstm_folder = "./logs/lstm/"
    vitlstm_folder = "./logs/vitlstm/"
    conv_folder = "./logs/convnet/"
    #expert_folder = "./logs/expert"

    #Spheres
    # vit_folder = "./logs_modified_md/vit/"
    # unet_folder = "./logs_modified_md/unet/"
    # lstm_folder = "./logs_modified_md/lstm/"
    # vitlstm_folder = "./logs_modified_md/vitlstm/"
    # conv_folder = "./logs_modified_md/convnet/"
    # expert_folder = "./logs_modified_md/expert"


    vit_traj_folders = sorted(glob.glob(opj(vit_folder, "*")))
    unet_traj_folders = sorted(glob.glob(opj(unet_folder, "*")))
    lstm_traj_folders = sorted(glob.glob(opj(lstm_folder, "*")))
    vit_lstm_traj_folders = sorted(glob.glob(opj(vitlstm_folder, "*")))
    conv_traj_folders = sorted(glob.glob(opj(conv_folder, "*")))
    #expert_traj_folders = sorted(glob.glob(opj(expert_folder, "*")))

    desiredVel=[3,4,5,6,7]
    #desiredVel=[3,3.5,4,4.5,5,5.5,6,6.5,7]
    vit = metrics(vit_traj_folders, desiredVel)
    unet = metrics(unet_traj_folders, desiredVel)
    lstm = metrics(lstm_traj_folders, desiredVel)
    vitlstm = metrics(vit_lstm_traj_folders, desiredVel)
    convnet = metrics(conv_traj_folders, desiredVel)
    
    #expert = metrics(expert_traj_folders, desiredVel)
    
    
    if 1:
        whatToPlot = 0
        titles = ["Success Rates", "JerkCost", "Commanded Accleration"]
        functions = [plotIt_sr, plotIt_energy, plotIt]

        if whatToPlot == 0:
            vit_successrate = vit.successRate()
            unet_successrate = unet.successRate()
            lstm_successrate = lstm.successRate()
            vitlstm_successrate = vitlstm.successRate()
            convnet_successrate = convnet.successRate()
            #expert_successrate = expert.successRate()

        if whatToPlot == 1:
            vit_successrate = vit.jerkCost()
            unet_successrate = unet.jerkCost()
            lstm_successrate = lstm.jerkCost()
            vitlstm_successrate = vitlstm.jerkCost()
            convnet_successrate = convnet.jerkCost()
            #expert_successrate = expert.jerkCost()
        if whatToPlot == 2:
            vit_successrate = vit.commandedAccel()
            unet_successrate = unet.commandedAccel()
            lstm_successrate = lstm.commandedAccel()
            vitlstm_successrate = vitlstm.commandedAccel()
            convnet_successrate = convnet.commandedAccel()
            #expert_successrate = expert.commandedAccel()
        if whatToPlot == 3:
            vit_successrate = vit.totalDistance()
            unet_successrate = unet.totalDistance()
            lstm_successrate = lstm.totalDistance()
            vitlstm_successrate = vitlstm.totalDistance()
            convnet_successrate = convnet.totalDistance()
            expert_successrate = expert.totalDistance()
        if whatToPlot == 4:
            vit_successrate = vit.timeTaken()
            unet_successrate = unet.timeTaken()
            lstm_successrate = lstm.timeTaken()
            vitlstm_successrate = vitlstm.timeTaken()
            convnet_successrate = convnet.timeTaken()
            expert_successrate = expert.timeTaken()


        success_rates = {
        'ViT': vit_successrate,
        'Unet': unet_successrate,
        'LSTMnet': lstm_successrate,
        'ViT+LSTM': vitlstm_successrate,
        'Convnet': convnet_successrate,
         }
        #'Expert': expert_successrate}
        # import pickle
        # with open(f'{titles[whatToPlot]}_trees.pkl', 'wb') as f:
        #     pickle.dump(success_rates, f)


        fn = functions[whatToPlot]
        fn( success_rates, desiredVel, titles[whatToPlot])







if __name__ == '__main__':
    main()

