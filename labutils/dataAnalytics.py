import glob, os, sys, time
from os.path import join as opj
import numpy as np
import torch
import matplotlib.pyplot as plt

import getpass
uname = getpass.getuser()

sys.path.append(f'/home/{uname}/agile_ws/src/agile_flight/learner')
sys.path.append(f'/home/{uname}/agile_ws/src/agile_flight/envtest/ros')
from learner_lstm import LearnerLSTM

dataset = sys.argv[1]

LearnerLSTM = LearnerLSTM(dataset_name=dataset, short=False, no_model=True)

#########################
## Velocity cmds value ##
#########################

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Full dataset vel cmds')

max_abs_val_vel = max(torch.abs(LearnerLSTM.train_velcmd[:,1:2]).max(), torch.abs(LearnerLSTM.val_velcmd[:,1:2]).max())

axs[0, 0].hist(LearnerLSTM.train_velcmd[:,0], bins=np.arange(2, 8, .5))
axs[0, 1].hist(LearnerLSTM.train_velcmd[:,1], bins=np.arange(-max_abs_val_vel, max_abs_val_vel, .05))
axs[0, 2].hist(LearnerLSTM.train_velcmd[:,2], bins=np.arange(-max_abs_val_vel, max_abs_val_vel, .05))
axs[0, 0].set_title('x cmd vel (train)')
axs[0, 1].set_title('y cmd vel (train)')
axs[0, 2].set_title('z cmd vel (train)')

axs[1, 0].hist(LearnerLSTM.val_velcmd[:,0], bins=np.arange(2, 8, .5))
axs[1, 1].hist(LearnerLSTM.val_velcmd[:,1], bins=np.arange(-max_abs_val_vel, max_abs_val_vel, .05))
axs[1, 2].hist(LearnerLSTM.val_velcmd[:,2], bins=np.arange(-max_abs_val_vel, max_abs_val_vel, .05))
axs[1, 0].set_title('x cmd vel (val)')
axs[1, 1].set_title('y cmd vel (val)')
axs[1, 2].set_title('z cmd vel (val)')

plt.subplots_adjust(hspace=0.2)

fig.savefig('vel_hist.png')

################################
## Mean velocity cmds per run ##
################################

# traj_idxs_train = np.hstack((0., np.cumsum(LearnerLSTM.train_trajlength)))






# fig, axs = plt.subplots(2, 3, figsize=(12, 8))
# fig.suptitle('Full dataset vel cmds')

# max_abs_val_vel = max(torch.abs(LearnerLSTM.train_velcmd[:,1:2]).max(), torch.abs(LearnerLSTM.val_velcmd[:,1:2]).max())

# axs[0, 0].hist(LearnerLSTM.train_velcmd[:,0], bins=np.arange(2, 8, .5))
# axs[0, 1].hist(LearnerLSTM.train_velcmd[:,1], bins=np.arange(-max_abs_val_vel, max_abs_val_vel, .05))
# axs[0, 2].hist(LearnerLSTM.train_velcmd[:,2], bins=np.arange(-max_abs_val_vel, max_abs_val_vel, .05))
# axs[0, 0].set_title('x cmd vel (train)')
# axs[0, 1].set_title('y cmd vel (train)')
# axs[0, 2].set_title('z cmd vel (train)')

# axs[1, 0].hist(LearnerLSTM.val_velcmd[:,0], bins=np.arange(2, 8, .5))
# axs[1, 1].hist(LearnerLSTM.val_velcmd[:,1], bins=np.arange(-max_abs_val_vel, max_abs_val_vel, .05))
# axs[1, 2].hist(LearnerLSTM.val_velcmd[:,2], bins=np.arange(-max_abs_val_vel, max_abs_val_vel, .05))
# axs[1, 0].set_title('x cmd vel (val)')
# axs[1, 1].set_title('y cmd vel (val)')
# axs[1, 2].set_title('z cmd vel (val)')

# plt.subplots_adjust(hspace=0.2)

# fig.savefig('vel_hist.png')













#################
## Depth image ##
#################

mean_im_train = LearnerLSTM.train_ims.mean(axis=0)
mean_im_val = LearnerLSTM.val_ims.mean(axis=0)

fig, axs = plt.subplots(1, 2, figsize=(8, 5))
fig.suptitle('Mean depth image observed')

im0 = axs[0].imshow(mean_im_train)
im1 = axs[1].imshow(mean_im_val)
axs[0].set_title('mean depth image (train)')
axs[1].set_title('mean depth image (val)')
fig.colorbar(im0, ax=axs[0])
fig.colorbar(im1, ax=axs[1])

fig.savefig('mean_depth_im.png')


