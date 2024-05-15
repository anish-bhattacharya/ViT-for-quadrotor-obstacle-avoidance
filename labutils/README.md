# Parameters

This readme is focused on how the parameters are to be changed. Since the configs are scattered in the whole repository, it is time to start keep track of the changes. 

## Trial Parameters
For each trial (simulation run), the parameters can be specified. You can choose easy, medium and hard difficulty of the environment. Moreover the environment itself can be specified. This config file is provided in
```flightmare/flightpy/configs/vision/config.yaml ```

## Obstacle settings
If you want to change the static obstacles or dynamic obstacles, you will require to go in the following folder

```flightmare/flightpy/configs/vision``` 

Inside the folder, there are obstacles for each map. In sub folder of difficulty, you can find static_obstacles.csv and dynamic_obstacles.yaml. The dynamic obstacles have trajectories defined in the traj folder. 

### Structure of Static Obstacles
The file by itself is a bit confusing so the structure is as follows.

|box name| x|y|z | q0 | q1| q2| q3| scale x| scale y| scale z|
| :---        |    :----:   |    :----:   |    :----:   |    :----:   |:----:   |:----:   |    :----:   |    :----:   |    :----:   |          ---: |


Reference: ```flightmare/flightlib/src/envs/vision_env/vision_env.cpp```
There will be a function <i>configStaticObjects</i> that uses the file. You can also refer to pathAnalysis.py for the usage.


## Reference Trajectory 

When you need to change the reference trajectory that is generated or maybe takeoff height, frequency of telemetry or the vehicle itself, go to

```envsim/parameters/simple_sim_pilot.yaml```


## Pipleine for Obstacle Generation

We will first generate our yaml file. This file will contain dummy positions but will have non varying spaces. 


