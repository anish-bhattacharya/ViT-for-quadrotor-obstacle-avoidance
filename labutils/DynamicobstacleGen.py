"""
The script generates random dynamic obstacles in the path of the drone
@V1.0
Date: 19th April 2023
@usage: python3 labutils/DynamicobstacleGen.py -n 10 (num obstacles)
@author: Dhruv Parikh
"""
import os
import numpy as np
import csv
import sys
import argparse
from utils import KR_agileUtils

spread = 5.0 # spread of the obstacles in y and z

def ObsGenToFile(_obsFile, numRuns, numObs,obstacle_base_position,obstacle_base_size) -> None:
    y = np.random.uniform(low=-spread/2, high=spread/2, size=(numRuns,numObs))
    z = np.random.uniform(low=-spread/2, high=spread/2, size=(numRuns,numObs))
    for i in range(numObs):
        obsFile = os.path.join(_obsFile,f"static_kr_{int(i)}.csv")
        with open(obsFile, "w") as csvfile:
            csvwriter = csv.writer(csvfile)

            for j in range(numRuns):
                obstaclePositions = [obstacle_base_position[i,0]+KR_agileUtils.randomGenBalanced(1.0),obstacle_base_position[i,1]+y[j,i],obstacle_base_position[i,2]+z[j,i],float(obstacle_base_size+KR_agileUtils.randomGenBalanced(1.0))]
                csvwriter.writerow(obstaclePositions)

    print("[Generator] Obstacles Generated!")


def readObstacles(ObsFile, lineNo, numObs) -> None:
    with open(ObsFile, "r") as file:
        csvreader = csv.reader(file)
        for itr, row in enumerate(csvreader):
            if itr == lineNo:
                y = [float(row[i]) for i in range(numObs)]
                z = [float(row[i]) for i in range(numObs, numObs*2)]
                return y, z
    return y, z

def generateDynamicObstacleConfigurationText(numObstacles, yamlFile):
    """
    If you touch this function it will break ->>space sensitive
    """

    text=f"""N: {int(numObstacles)}
Object1:
  csvtraj: traj_3885698445451241647
  loop: false
  position:
  - 34.434292120817815
  - 2.465347946075196
  - 9.196069921085115
  prefab: rpg_box01
  rotation:
  - 0.24704812646862423
  - 0.9327309480410784
  - -0.2131693666866219
  - 0.15342432297767733
  scale:
  - 0.7914929684780745
  - 0.7914929684780745
  - 0.7914929684780745\n"""
    for i in range(numObstacles):
        text += f"""Object{int(i+2)}:\n  csvtraj: traj_3885698445451241647\n  loop: false
  position:
  - 4.434292120817815\n  - 0.465347946075196\n  - 10.196069921085115\n  prefab: rpg_box01
  rotation:
  - 0.24704812646862423\n  - 0.9327309480410784\n  - -0.2131693666866219\n  - 0.15342432297767733
  scale:
  - 3.7914929684780745\n  - 3.7914929684780745\n  - 3.7914929684780745\n"""
    utility = KR_agileUtils()
    utility.rewriteFile(yamlFile, text)

def singleObsGen() -> None:
    ####################
    # Argument Parsers #
    ####################

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-n", "--numRuns", help="Pass number of runs")

    args = argParser.parse_args()
    if len(sys.argv) < 1:
        argParser.print_help()
        argParser.exit()
    numRuns = int(args.numRuns)
    
    ####################
    # Dummy obstacles  #
    ####################

    yamlFile = "flightmare/flightpy/configs/vision/hard/environment_0/dynamic_obstacles.yaml"
    #I am keeping the num Obstacles hardcoded
    numObstacles = 13
    generateDynamicObstacleConfigurationText(numObstacles,yamlFile)
    
    ####################
    # Obs Generation   #
    ####################

    dynamicAbsolutePosition = "flightmare/flightpy/configs/vision/hard/environment_0/"
    """
    We will make the following configuration
    at z = 15 meters
      x 
    x x x
      x

    at z = 12 meters    
    x        x 

    at z = 23 meters
    x        x     
    """
    # 13 obstacles
    obstacle_base_position = np.array([[15,0,3],
                                       [15,-1.5,3],
                                       [15,1.5,3],
                                       [15,0,4.5],
                                       [15,0,1.5],
                                       [12,-2,3],
                                       [12,2,3],
                                       [17,2,3],
                                       [17,-2,3],
                                       [23,0,3],
                                       [23,-1,3],
                                       [23,1,1],
                                       [23,0,5]])

    # five obstacles
    # obstacle_base_position = np.array([
    #                                    [10, 0, 3],
    #                                    [10, 0, 3],
    #                                    [11, 0, 3],
    #                                    [12, 0, 3],
    #                                    [12, 0, 3],
    #                                    [13, 0, 3],
    #                                    [14, 0, 3],
    #                                    [14, 0, 3],
    #                                    [15, 0, 3],
    #                                    [12, 0, 3],
    #                                    [13, 0, 3],
    #                                    [14, 0, 3],
    #                                    [14, 0, 3],
    #                                    [15, 0, 3],
    #                                    [15, 0, 3],
    #                                    [16, 0, 3],
    #                                    [17, 0, 3],
    #                                    [18, 0, 3],
    #                                    [19, 0, 3],
    #                                    [18, 0, 3],
    #                                    [18, 0, 3],
    #                                    [19, 0, 3],
    #                                    [20, 0, 3],
    #                                    [20, 0, 3],
    #                                    [21, 0, 3],
    #                                    [21, 0, 3],
    #                                    [21, 0, 3],
    #                                    [22, 0, 3],
    #                                    [22, 0, 3],
    #                                    [23, 0, 3],
    #                                    [23, 0, 3],
    #                                    [24, 0, 3],
    #                                    [24, 0, 3],
    #                                    [25, 0, 3],
    #                                    [25, 0, 3],
    #                                    ])

    obstacle_base_size = 1.5

    print(f"[Generator] Generating obstacles for {args.numRuns} trials")
    ObsGenToFile(dynamicAbsolutePosition, numRuns, numObstacles, obstacle_base_position, obstacle_base_size)
    print(f"[Generator] {args.numRuns} Obstacles Map Generated!")


if __name__=="__main__":
    singleObsGen()