#!/bin/bash

# Pass number of rollouts as argument
if [ $1 ]
then
  N="$1"
else
  N=5
fi

echo $2

if [ "$2" = "vision" ]
then
  echo
  echo "[LAUNCH SCRIPT] Vision based!"
  echo
  run_competition_args="--vision_based --model_type $3 --model_path $4 --num_lstm_layers $5"
  realtimefactor=""
elif [ "$2" = "state" ]
then
  echo
  echo "[LAUNCH SCRIPT] State based!"
  echo
  if [ "$3" = "human" ]
  then
    run_competition_args="--keyboard"
    realtimefactor="real_time_factor:=1.0"
  else
    run_competition_args=""
    realtimefactor="real_time_factor:=1.0"
  fi
else
  echo
  echo "[LAUNCH SCRIPT] Unknown or empty second argument: $2, only 'vision' or 'state' allowed!"
  echo
  exit 1
fi

# Set Flightmare Path if it is not set
if [ -z $FLIGHTMARE_PATH ]
then
  export FLIGHTMARE_PATH=$PWD/flightmare
fi

# Launch the simulator, unless it is already running
if [ -z $(pgrep visionsim_node) ]
then
  roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=True $realtimefactor &
  ROS_PID="$!"
  echo $ROS_PID
  sleep 10
else
  ROS_PID=""
fi

# Perform N evaluation runs
for i in $(eval echo {1..$N})
do
  # Publish simulator reset
  rostopic pub /kingfisher/dodgeros_pilot/off std_msgs/Empty "{}" --once
  rostopic pub /kingfisher/dodgeros_pilot/reset_sim std_msgs/Empty "{}" --once
  rostopic pub /kingfisher/dodgeros_pilot/enable std_msgs/Bool "data: true" --once
  rostopic pub /kingfisher/dodgeros_pilot/start std_msgs/Empty "{}" --once
  
  export ROLLOUT_NAME="rollout_""$i"
  echo "$ROLLOUT_NAME"
​
  cd ./envtest/ros/
  python3 evaluation_node.py ${datetime}_N$i &
  PY_PID="$!"
​
  python3 run_competition.py $run_competition_args &
  COMP_PID="$!"
​
  cd -
​
  sleep 2.0
  rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" --once
​
  # Wait until the evaluation script has finished
  while ps -p $PY_PID > /dev/null
  do
    sleep 1
  done
​
  cat "$SUMMARY_FILE" "./envtest/ros/summary.yaml" > "tmp.yaml"
  mv "tmp.yaml" "$SUMMARY_FILE"
​
  # kill -SIGINT "$COMP_PID"
done
​
​
if [ $ROS_PID ]
then
  # kill -SIGINT "$ROS_PID"
fi
