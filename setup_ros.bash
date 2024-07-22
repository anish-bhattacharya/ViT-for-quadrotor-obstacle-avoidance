#!/bin/bash

if [[ ! -f "$(pwd)/setup_ros.bash" ]]
then
  echo "please launch from the repository folder!"
  exit
fi

project_path=$PWD
echo $project_path

echo "Using apt to install dependencies..."
echo "Will ask for sudo permissions:"
sudo apt update
sudo apt install -y --no-install-recommends build-essential cmake libzmqpp-dev libopencv-dev unzip python3-catkin-tools
sudo pip install uniplot

echo "Ignoring unused Flightmare folders!"
touch flightmare/flightros/CATKIN_IGNORE

echo "Setting the flightmare environment variable. Please add 'export FLIGHTMARE_PATH=$PWD/flightmare' to your .bashrc!"
export FLIGHTMARE_PATH=$project_path/flightmare

echo "Creating envtest/ros/train_set folder to store images and telemetry during flights..."
mkdir $project_path/envtest/ros/train_set

echo "Done!"
echo "Have a save flight!"
