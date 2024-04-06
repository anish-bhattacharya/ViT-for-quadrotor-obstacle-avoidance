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
  run_competition_args="--vision_based"
  realtimefactor=""
elif [ "$2" = "state" ]
then
  echo
  echo "[LAUNCH SCRIPT] State based!"
  echo
  run_competition_args="--state_based"
  if [ "$3" = "human" ]
  then
    run_competition_args="--keyboard"
    realtimefactor="real_time_factor:=1.0"
  else
    run_competition_args=""
    realtimefactor="real_time_factor:=10.0"
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

# env_list=("13" "30" "48" "73" "89" "7" "16" "67" "57" "42" "72")
env_list=("73")
for env_l in ${!env_list[@]}
do
  #Creating helper folders (empty) to track which folders in train_set corresponds to which setting
  mkdir "/home/dhruv/icra22_competition_ws/src/agile_flight/envtest/ros/train_set/""${env_list[$env_l]}""_start"
  echo "================= LOADING ENV ${env_list[$env_l]} =================== "
  python3 /home/dhruv/icra22_competition_ws/src/agile_flight/modify_env.py --env_n ${env_list[$env_l]}
  # sleep 30
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

  SUMMARY_FILE="evaluation.yaml"
  echo "" > $SUMMARY_FILE

  # generate datetime string to label summary folders with in evaluation_node.py
  datetime=$(date '+d%m_%d_t%H_%M')

  relaunch_sim=0

  # Perform N evaluation runs
  # des_vel_list=("3" "3.5" "4" "4.5" "5" "5.5" "6" "6.5" "7")
  des_vel_list=("5")
  models=("LSTMNet" "UNetLSTM" "ConvNet" "vit" "vitlstm")
  # models_path=("/home/dhruv/Desktop/nometadata/lstm/model_000075.pth" "/home/dhruv/Desktop/nometadata/unet/model_000075.pth" "/home/dhruv/Desktop/nometadata/conv/model_000075.pth" "/home/dhruv/Desktop/nometadata/vit/model_000075.pth" "/home/dhruv/Desktop/nometadata/vitlstm/model_000075.pth")
  # # models_path=("/home/dhruv/tests/colmodels/vitlstm/model_75.pth")
  # models=("LSTMNet")
  models_path=("/home/dhruv/tests/metadatafix/lstm/model_000075.pth" "/home/dhruv/tests/metadatafix/unet/model_000075.pth" "/home/dhruv/tests/colmodels/convnet/model.pth" "/home/dhruv/tests/colmodels/vit/model_75.pth" "/home/dhruv/tests/colmodels/vitlstm/model_75.pth")



  for model_k in ${!models[@]}
  do
    mkdir "/home/dhruv/icra22_competition_ws/src/agile_flight/envtest/ros/train_set/""${models[$model_k]}""_""${env_list[$env_l]}""_""_start"
    for des_vel_j in ${!des_vel_list[@]}
    do
      for i in $(eval echo {1..$N})
      do
        # Reset the simulator if needed
        if ((relaunch_sim))
        then
            echo
            echo
            echo
            echo
            echo RELAUNCHING SIMULATOR ON RUN $i
            echo
            echo
            echo
            echo

          # reset flag and kill everything to restart
          relaunch_sim=0
          killall -9 roscore rosmaster rosout gzserver gzclient RPG_Flightmare.
          sleep 10

          # Launch the simulator, unless it is already running
          if [ -z $(pgrep visionsim_node) ]
          then
            roslaunch envsim visionenv_sim.launch render:=True gui:=False rviz:=True $realtimefactor &
            ROS_PID="$!"
            echo $ROS_PID
            sleep 10
          else
            killall -9 roscore rosmaster rosout gzserver gzclient RPG_Flightmare.
            sleep 10
          fi

        fi

        start_time=$(date +%s)

        # Publish simulator reset
        rostopic pub /kingfisher/dodgeros_pilot/off std_msgs/Empty "{}" --once
        rostopic pub /kingfisher/dodgeros_pilot/reset_sim std_msgs/Empty "{}" --once
        rostopic pub /kingfisher/dodgeros_pilot/enable std_msgs/Bool "data: true" --once
        rostopic pub /kingfisher/dodgeros_pilot/start std_msgs/Empty "{}" --once

        export ROLLOUT_NAME="rollout_""$i"
        echo "$ROLLOUT_NAME"

        cd ./envtest/ros/
        python3 evaluation_node.py ${datetime}_N$i &
        PY_PID="$!"
        #python3 /home/dhruv/icra22_competition_ws/src/agile_flight/labutils/desiredVelGen.py -f 0 -n $i

        python3 run_competition.py $run_competition_args --num_lstm_layers ${des_vel_list[$des_vel_j]} --model_type "LSTMNet" --model_path ${models_path[$model_k]}&
        # python3 run_competition.py $run_competition_args --num_lstm_layers ${des_vel_list[$des_vel_j]} --model_type ${models[$model_k]} --model_path ${models_path[$model_k]}&
        COMP_PID="$!"

        cd -

        sleep 2

        # Wait until the evaluation script has finished
        while ps -p $PY_PID > /dev/null
        do
          echo
          echo [LAUNCH_EVALUATION] Sending start navigation command
          echo
          rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" --once
          sleep 2

          # if the current iteration has surpassed the time limit, something went wrong (possibly: [Pipeline]     Bridge failed!). Kill the simulator.
          if ((($(date +%s) - start_time) >= 300))
          then
            echo
            echo
            echo
            echo
            echo "Time limit exceeded. Exiting evaluation script loop."
            echo
            echo
            echo
            echo
            kill -SIGINT $PY_PID
            relaunch_sim=1
            break
          fi

        done

        cat "$SUMMARY_FILE" "./envtest/ros/summary.yaml" > "tmp.yaml"
        mv "tmp.yaml" "$SUMMARY_FILE"

        kill -SIGINT "$COMP_PID"
      done
    
    done
    mkdir "/home/dhruv/icra22_competition_ws/src/agile_flight/envtest/ros/train_set/""${models[$model_k]}""_""${env_list[$env_l]}""_""_end"
  done

if [ $ROS_PID ]
then
  kill -SIGINT "$ROS_PID"
fi
mkdir "/home/dhruv/icra22_competition_ws/src/agile_flight/envtest/ros/train_set/""${env_list[$env_l]}""_end"
done
