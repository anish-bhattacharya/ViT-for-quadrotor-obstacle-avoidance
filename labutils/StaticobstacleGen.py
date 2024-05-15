"""
The script generates random static obstacle in the path of the drone
@V2.0
Date: 30th Jan 2023
@usage: python3 StaticobtsacleGen.py > out.txt
Copy the out.txt and paste into the static_obstacles.csv
"""
from random import random
import random
import os
import numpy as np
import csv
import sys
import argparse

def randomGenBalanced(ran=1):
        """
        Balanced Random Number Generator -> add to utils class in future
        """
        return (random.random() - 0.5) * ran
# Static obstacle.csv sample data
# box name, x,y,z , q0, q1, q2, q3, scale x, scale y, scale z
# rpg_box01, 12.21092254463925, -1.3311644287791111, 2.896233998379233, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 0.2029075814850837, 0.2029075814850837, 0.2029075814850837

# 10-17
# 7 meters and 10 obstacles in a row

# def ObsGenToFile(obsFile,numRuns)->None:
#     y = np.random.uniform(low=-0.7, high=0.7, size=(numRuns,3))
#     z = np.random.uniform(low=-0.5, high=0.8, size=(numRuns,3))

def ObsGenToFile(obsFile, numRuns) -> None:
    y = np.random.uniform(low=-0.7, high=0.7, size=(numRuns,3))
    z = np.random.uniform(low=-0.5, high=0.8, size=(numRuns,3))

    with open(obsFile, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        for i in range(numRuns):
            csvwriter.writerow([y[i, 0], y[i, 1], y[i, 2], z[i, 0], z[i, 1], z[i, 2]])

    print("[Generator] Obstacles Generated!")


def readObstacles(ObsFile, lineNo) -> None:
    with open(ObsFile, "r") as file:
        csvreader = csv.reader(file)
        for itr, row in enumerate(csvreader):
            if itr == lineNo:
                y = [float(row[i]) for i in range(3)]
                z = [float(row[i]) for i in range(3, 6)]
                return y, z
    return y, z





def selectObsMode(mode):
    # 5 modes
    j = 15
    # Designed in such a way that
    # middle is 1-3
    # down is 4-6
    # up is 7-10
    obsMaster = [
        f"rpg_box01, {j},{0.079374539622437}, {2.0}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1, 1, 1",
        f"rpg_box01, {j},{0}, {3}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 2,2,2",
        f"rpg_box01, {j},{-1.5+randomGenBalanced(1)}, {3+randomGenBalanced(1)}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1.5,1.5,1.5",
        f"rpg_box01, {j},{1.5+randomGenBalanced(1)}, {3+randomGenBalanced(1)}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1.5,1.5,1.5",
        f"rpg_box01, {j},{0}, {1.5}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1.5,1.5,1.5",
        f"rpg_box01, {j},{-1.5+randomGenBalanced(1)}, {1.5+randomGenBalanced(1)}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1.5,1.5,1.5",
        f"rpg_box01, {j},{1.5+randomGenBalanced(1)}, {1.5+randomGenBalanced(1)}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1.5,1.5,1.5",
        f"rpg_box01, {j},{0}, {4.5}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1.5,1.5,1.5",
        f"rpg_box01, {j},{-1.5+randomGenBalanced(1)}, {4.5+randomGenBalanced(1)}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1.5,1.5,1.5",
        f"rpg_box01, {j},{1.5+randomGenBalanced(1)}, {4.5+randomGenBalanced(1)}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1.5,1.5,1.5",
    ]

    # Up
    if mode == 1:
        obsMaster.pop(-3)
    # Down
    elif mode == 2:
        obsMaster.pop(1)
    # Right
    elif mode == 3:
        obsMaster.pop(2)
    # Left
    elif mode == 4:
        obsMaster.pop(3)
    elif mode == 5:
        obsMaster.pop(4)

    return obsMaster


def artificalObsMode(obsPath, mode) -> None:
    """
    Aritifical Obstacle Mode
    @params: File of obstacle
    Mode of obstacle
    """
    if mode not in [1, 2, 3, 4, 5]:
        print(
            "[Generator] Bad Mode Specified. For your Punishment, I will generate a wall!"
        )

    with open(obsPath, "w") as f:
        obs = selectObsMode(mode)
        itr = 0
        for i in obs:
            itr = itr + 1
            f.write(f"{str(i)}\n")  # Making sure we are entering strings


def singleObsGen() -> None:
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-o", "--genObs", help="Pass 1 if generate new set of Obstacles"
    )
    argParser.add_argument("-n", "--numRuns", help="Pass number of runs")
    argParser.add_argument("-c", "--curRun", help="Pass Current run")
    argParser.add_argument("-m", "--mode", help="Mode for Obstacles")
    argParser.add_argument("-w", "--world", help="World Generator")

    args = argParser.parse_args()
    if len(sys.argv) < 2:
        argParser.print_help()
        argParser.exit()
    obsPath = "flightmare/flightpy/configs/vision/hard/environment_0/static_obstacles.csv"
    if int(args.world) == 1:
        print("[Generator] Generating Map of obstacles")
        x, y, z = wholeMap()
        with open(obsPath, "w") as f:
            for i in range(len(x)):

                obstacle = f'rpg_box01, {x[i]},{y[i]}, {z[i]}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1.6, 1.6, 1.6'
                #Write twice for first obstacle 
                if i==0:            
                    f.write(f"{str(obstacle)}\n")
                f.write(f"{str(obstacle)}\n")
        sys.exit(0)

    obsFile = "obstacles.csv"

    lineNo = int(args.curRun)

    print(f"[Generator] Run No.: {lineNo}")

    if int(args.genObs):
        print(f"[Generator] Generating obstacles for {args.numRuns} trials")
        numRuns = int(args.numRuns)
        ObsGenToFile(obsFile, numRuns)

    if int(args.mode) != 0:
        print(f"[Generator] Generating Artifical Obstacles")
        artificalObsMode(obsPath, int(args.mode))
        sys.exit(0)

    



    y,z = readObstacles(obsFile,lineNo)
    with open(obsPath, 'w') as f:
        j = 15 +randomGenBalanced(0.4) 
        scale = randomGenBalanced(1) + 1.9
        obs = [f'rpg_box01, {j},{0.079374539622437+y[0]}, {2.0+z[0]}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1, 1, 1',
        f'rpg_box01, {j+randomGenBalanced(0.5)},{-1.2 + y[0]}, {3+z[0]}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1.85, 1.85, 1.85',
        f'rpg_box01, {j},{0.00+y[1]}, {3+z[1]}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, {scale},{scale},{scale}',
        f'rpg_box01, {j+randomGenBalanced(0.5)},{1.2+y[2]}, {3+z[2]}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1.85, 1.85, 1.85',
        f'rpg_box01, {j+1},{0.00+y[1]}, {4.5+randomGenBalanced(1)}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1.85, 1.85, 1.85',
        f'rpg_box01, {j+1},{0.00+y[1]}, {1+randomGenBalanced(0.5)}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1.85, 1.85, 1.85',
        f'rpg_box01, {j-1},{-2.8+y[2]}, {3+z[2]+randomGenBalanced(2)}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1.85, 1.85, 1.85',
        f'rpg_box01, {j-2},{3+y[1]}, {3+z[0]+randomGenBalanced(2)}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1.85, 1.85, 1.85',
        
        
        f'rpg_box01, {j+5},{0.079374539622437}, {2.0}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1, 1, 1',
        f'rpg_box01, {j+5},{0}, {3}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 2,2,2',
        f'rpg_box01, {j+5},{-1.5+randomGenBalanced(3)}, {3+randomGenBalanced(3)}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1.5,1.5,1.5',
        f'rpg_box01, {j+5},{1.5+randomGenBalanced(3)}, {3+randomGenBalanced(3)}, 0.05119733996721293, -0.6597787329839699, -0.10994562703289923, 0.7416082624800586, 1.5,1.5,1.5']

        itr = 0
        for i in obs:
            itr = itr + 1
            f.write(f"{str(i)}\n")  # Making sure we are entering strings


def wholeMap()->None:    
    #This time we can generate obstacles on each run because of the density 
    Y = np.random.normal(0, .75, size=(1,4))
    y = np.random.uniform(low=-5,high=2 ,size=(1,15))
    y = np.column_stack((y,Y))


    y = list(y.flatten())
    for i in range(10):
        random.shuffle(y)
    y = np.array(y)
    
    x = np.random.uniform(low=10,high=17,size=(1,4))
    X = np.random.uniform(low=12,high=17,size=(1,15))
    x = np.column_stack((x,X))
    x = list(x.flatten())
    random.shuffle(x)
    x = np.array(x)

    Z = np.random.normal(3, 1, size=(1,4))
    z = np.random.uniform(low=.5,high=5,size=(1,15))
    z = np.column_stack((z,Z))


    print("[Generator] Obstacles Generated")
    return x.flatten(), y.flatten(), z.flatten()


if __name__ == "__main__":
    singleObsGen()
