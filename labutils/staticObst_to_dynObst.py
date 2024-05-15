# we need to convert the given static_obstacles.csv files from each environment folder into the static_kr_{i}.csv files to be used by our dynamic obstacle moving code.

import os
import csv
from utils import KR_agileUtils
import numpy as np

tmp = KR_agileUtils()

# Define the number of rows in the original CSV files
N = 468
output_directory = 'medium'

base_directory = '/home/dhruv/icra22_competition_ws/src/agile_flight/flightmare/flightpy/configs/vision/'

# # Loop through each row
for row in range(N):
    # break
    
    # Create a new CSV file for each row
    output_file = os.path.join(base_directory, output_directory, "custom_"+output_directory, f"static_kr_{row}.csv")
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        X,Y,Z = tmp.randomGenBalanced(4),tmp.randomGenBalanced(4),tmp.randomGenBalanced(4)
        # Loop through each environment folder
        for environment in range(1, 101):
            # Read the original CSV file
            input_file = os.path.join(base_directory, output_directory, f"environment_73/static_obstacles.csv")
            with open(input_file, mode='r') as input_csv:
                y = np.random.uniform(low=-6,high=6)
                x = np.random.uniform(low=10,high=40)
                z = np.random.uniform(low=1,high=7)
                reader = csv.reader(input_csv)
                
                # Extract the desired elements from the row
                extracted_row = list(reader)[row]
                desired_info = [X+np.clip(float(extracted_row[1]),5,None), Y+float(extracted_row[2]), Z+float(extracted_row[3]),np.clip(float( extracted_row[8])-0.2,0.8,None)]
                # desired_info = [x, 
                #                 y, 
                #                 z,
                #                 np.clip(np.abs(tmp.randomGenBalanced(4)),1.2,None)]
                desired_info = [str(i) for i in desired_info]
                
                # Write the extracted row to the new CSV file
                writer.writerow(desired_info)

# write yaml file
def basetext(n):
    basetext = \
f'''
    csvtraj: traj_37158987063224946
    loop: false
    position:
    - 48.057093511629645
    - 3.4678879517038
    - 7.923471482431094
    prefab: rpg_box0{n}
    rotation:
    - 0.4846908786270495
    - -0.35626152747714557
    - -0.02281735819161093
    - 0.7985185310188774
    scale:
    - 1.5386464779409343
    - 1.5386464779409343
    - 1.5386464779409343
'''
    return basetext
filename = os.path.join(base_directory, output_directory, "custom_"+output_directory, "dynamic_obstacles.yaml")
with open(filename, mode='w') as file:
    file.write(f"N: {N}\n")
for o in range(N):
    with open(filename, mode='a') as file:
        n = 1 #np.random.choice([1,2,3])
        file.write(f"Object{o+1}:{basetext(n)}")
