#!/bin/bash

# Provide number of environments to be generated

if [ -z $1 ]
then
  N=10
else
  N=$1
fi

rm -rf environment*

for i in $(eval echo {0..$N})
do
  dirname="environment_""$i"
  mkdir -p $dirname
  python3 obstacle_generator.py "$RANDOM"
  mv csvtrajs $dirname
  mv dynamic_obstacles.yaml $dirname
  mv static_obstacles.csv $dirname
done
