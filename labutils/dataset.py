from re import S
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


dataDir = "/media/dhruv/New Volume/agile_flight/100FOV"

#List of all folders in the trianset folder
allFolders = [os.path.join(dataDir, i) for i in os.listdir(dataDir)]


velX = []
velY = []
velZ= []
for folder in sorted(allFolders):
    try:
        data = pd.read_csv(folder+"/data.csv")
    except:
        continue
    velX += list(data['velcmd_x'].to_numpy())
    velY += list(data['velcmd_y'].to_numpy())
    velZ += list(data['velcmd_z'].to_numpy())

plt.hist(velZ)    
plt.show()


# ffmpeg -r 90 -i simplescreenrecorder-2024-02-03_09.39.02.mkv  -vf "setpts=0.5*PTS" vid.gif 