#pragma once


#include <dynamic_reconfigure/server.h>
#include <ros/ros.h>

#include <condition_variable>
#include <mutex>

#include "dodgelib/math/math.hpp"
#include "dodgelib/pilot/pilot.hpp"
#include "dodgelib/sampler/time_based/time_sampler.hpp"
#include "dodgeros/bridge/ros_bridge.hpp"
#include "dodgeros/ros_setpoint.hpp"
#include "dodgeros/ros_traj_visualizer.hpp"
#include "dodgeros_msgs/Command.h"
#include "dodgeros_msgs/Reference.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/TwistStamped.h"
#include "mav_msgs/Actuators.h"
#include "nav_msgs/Odometry.h"
#include "sensor_msgs/BatteryState.h"
#include "sensor_msgs/Imu.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Empty.h"
#include "std_msgs/Float32.h"
#include "std_srvs/SetBool.h"
#include "std_srvs/Trigger.h"


namespace agi {

class RosPilot {
 public:
  RosPilot(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
  RosPilot() : RosPilot(ros::NodeHandle(), ros::NodeHandle("~")) {}
  ~RosPilot();
  Command getCommand() const;
  bool getQuadrotor(Quadrotor* const quad) const;
  Pilot& getPilot();

 private:
  void runPipeline(const ros::TimerEvent& event);
  void stateEstimateCallback(const dodgeros_msgs::QuadState& msg);
  void odometryEstimateCallback(const nav_msgs::OdometryConstPtr& msg);
  void poseEstimateCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  void imuCallback(const sensor_msgs::ImuConstPtr& msg);
  void motorSpeedCallback(const mav_msgs::Actuators& msg);
  void startCallback(const std_msgs::EmptyConstPtr& msg);
  void forceHoverCallback(const std_msgs::EmptyConstPtr& msg);
  void goToPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg);
  void velocityCallback(const geometry_msgs::TwistStampedConstPtr& msg);
  void landCallback(const std_msgs::EmptyConstPtr& msg);
  void offCallback(const std_msgs::EmptyConstPtr& msg);
  void enableCallback(const std_msgs::BoolConstPtr& msg);
  void trajectoryCallback(const dodgeros_msgs::ReferenceConstPtr& msg);
  void voltageCallback(const sensor_msgs::BatteryStateConstPtr& msg);
  void feedthroughCommandCallback(const dodgeros_msgs::CommandConstPtr& msg);

  void advertiseDebugVariable(const std::string& var_name);
  void publishDebugVariable(const std::string& name,
                            const PublishLogContainer& container);
  void publishLoggerDebugMsg();
  void pipelineCallback(const QuadState& state, const Feedback& feedback,
                        const ReferenceVector& references,
                        const SetpointVector& reference_setpoints,
                        const SetpointVector& outer_setpoints,
                        const SetpointVector& inner_setpoints,
                        const Command& command);

  void referencePublisher();
  Scalar updateRmse(const QuadState& state, const SetpointVector& setpoints,
                    const ReferenceVector& references) const;

  // ROS members
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber pose_estimate_sub_;
  ros::Subscriber odometry_estimate_sub_;
  ros::Subscriber state_estimate_sub_;
  ros::Subscriber imu_sub_;
  ros::Subscriber motor_speed_sub_;
  ros::Subscriber start_sub_;
  ros::Subscriber force_hover_sub_;
  ros::Subscriber go_to_pose_sub_;
  ros::Subscriber velocity_sub_;
  ros::Subscriber land_sub_;
  ros::Subscriber off_sub_;
  ros::Subscriber trajectory_sub_;
  ros::Subscriber enable_sub_;
  ros::Subscriber voltage_sub_;
  ros::Subscriber feedthrough_command_sub_;

  ros::Publisher state_pub_;
  ros::Publisher state_odometry_pub_;
  ros::Publisher telemetry_pub_;
  ros::Publisher cmd_pub_;

  ros::Timer run_pipeline_timer_;

  // Dodgelib
  PilotParams params_;
  Pilot pilot_;
  Logger logger_{"RosPilot"};

  std::unordered_map<std::string, ros::Publisher> logger_publishers_;

  // Trajectory visualization
  RosTrajVisualizer reference_visualizer_;
  RosTrajVisualizer active_reference_visualizer_;
  RosTrajVisualizer outer_setpoints_visualizer_;
  RosTrajVisualizer inner_setpoints_visualizer_;

  // Reference publishing
  bool shutdown_{false};
  std::thread reference_publishing_thread_;
  std::mutex reference_publishing_mtx_;
  std::condition_variable reference_publishing_cv_;
  std::vector<std::shared_ptr<ReferenceBase>> references_;
  Vector<4> motor_speeds_{NAN, NAN, NAN, NAN};
};

}  // namespace agi
