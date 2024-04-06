# ViT-for-quadrotor-obstacle-avoidance
Official repository for the paper "Utilizing vision transformer models for end-to-end vision-based quadrotor obstacle avoidance"  by Bhattacharya, et al. (2024) from GRASP, Penn.

# Build 
```
cd     # or wherever you'd like to install this code
export ROS_VERSION=noetic
export CATKIN_WS=./iros24_ws
mkdir -p $CATKIN_WS/src
cd $CATKIN_WS
catkin init
catkin config --extend /opt/ros/$ROS_VERSION
catkin config --merge-devel
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-fdiagnostics-color
cd src
git clone https://github.com/anish-bhattacharya/ViT-for-quadrotor-obstacle-avoidance/
cd ViT-for-quadrotor-obstacle-avoidance
./setup_ros.bash
catkin build
```
# Change Environments
Download the binary and put it in 
```
flightmare/flightrender/*.x86_64
```
Set the file name in `envsim/launch/visionenv_sim.launch`. 
Run the following commands
```
find . -name "CMakeCache.txt" -exec rm -rf {} +
find . -name "CMakeFiles" -type d -exec rm -rf {} +
catkin build
```
You are all set. Make sure you have the environments and configs updated!

# TODO
train_set, config and the output dirs have to be created before the first launch

