#pragma once

#include <yaml-cpp/yaml.h>

#include <memory>

// -- ros
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>

#include <filesystem>

#include "std_msgs/String.h"

// -- agilicious
#include "dodgelib/base/parameter_base.hpp"
#include "dodgelib/simulator/model_init.hpp"
#include "dodgelib/simulator/model_motor.hpp"
#include "dodgelib/simulator/model_rigid_body.hpp"
#include "dodgelib/simulator/model_thrust_torque_simple.hpp"
#include "dodgelib/simulator/quadrotor_simulator.hpp"
#include "dodgelib/utils/timer.hpp"
#include "dodgeros/ros_pilot.hpp"

// flightlib
#include "flightlib/envs/vision_env/vision_env.hpp"

namespace agi {

class VisionSim {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VisionSim(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
  VisionSim() : VisionSim(ros::NodeHandle(), ros::NodeHandle("~")) {}
  ~VisionSim();

 private:
  void resetCallback(const std_msgs::EmptyConstPtr& msg);

  void simLoop();
  void publishState(const QuadState& state);
  void publishImages(const QuadState& state);
  void publishObstacles(const QuadState& state);

  ros::NodeHandle nh_, pnh_;
  ros::Subscriber reset_sub_;
  ros::Publisher odometry_pub_;
  ros::Publisher state_pub_;
  ros::Publisher clock_pub_;

  ros::Publisher obstacle_pub_;

  image_transport::Publisher image_pub_;
  image_transport::Publisher depth_pub_;
  image_transport::Publisher opticalflow_pub_;

  Quadrotor quad_;
  QuadrotorSimulator simulator_;
  RosPilot ros_pilot_;
  Scalar camera_dt_ = 0.04;  // 20 Hz. Should be a multiple of sim_dt_
  Scalar sim_dt_ = 0.01;
  int render_every_n_steps_ = camera_dt_ / sim_dt_;
  int step_counter_ = 0;
  Scalar real_time_factor_ = 1.0;
  bool render_ = false;
  ros::WallTime t_start_;

  std::string agi_param_directory_;
  std::string ros_param_directory_;

  // flightmare vision environment
  std::unique_ptr<flightlib::VisionEnv> vision_env_ptr_;
  flightlib::FrameID frame_id_;

  // -- Race tracks
  Vector<3> start_pos_;
  Vector<3> goal_pos_;

  std::mutex sim_mutex_;
  std::thread sim_thread_;
  std::thread render_thread_;
};

}  // namespace agi