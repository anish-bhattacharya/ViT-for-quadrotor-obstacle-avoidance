#pragma once

#include <ros/ros.h>

#include "dodgelib/simulator/model_body_drag.hpp"
#include "dodgelib/simulator/model_init.hpp"
#include "dodgelib/simulator/model_motor.hpp"
#include "dodgelib/simulator/model_propeller_bem.hpp"
#include "dodgelib/simulator/model_rigid_body.hpp"
#include "dodgelib/simulator/model_thrust_torque_simple.hpp"
#include "dodgelib/simulator/quadrotor_simulator.hpp"
#include "dodgelib/utils/timer.hpp"
#include "dodgeros/ros_pilot.hpp"

namespace agi {

class AgiSim {
 public:
  AgiSim(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
  AgiSim() : AgiSim(ros::NodeHandle(), ros::NodeHandle("~")) {}
  ~AgiSim();

 private:
  void resetCallback(const std_msgs::BoolConstPtr& msg);
  void simLoop();
  void publishStates(const QuadState& state);

  ros::NodeHandle nh_, pnh_;
  ros::Subscriber reset_sub_;
  ros::Publisher odometry_publisher_;
  ros::Publisher state_publisher_;
  ros::Publisher clock_publisher_;

  Quadrotor quad_;
  QuadrotorSimulator simulator_;
  RosPilot ros_pilot_;
  Scalar sim_dt_ = 0.005;
  Scalar real_time_factor_ = 1.0;
  std::thread sim_thread_;
};

}  // namespace agi
