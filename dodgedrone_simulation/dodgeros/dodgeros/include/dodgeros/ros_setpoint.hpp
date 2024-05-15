#pragma once

#include "dodgelib/types/setpoint.hpp"
#include "dodgeros/ros_eigen.hpp"
#include "geometry_msgs/PoseStamped.h"

namespace agi {

class RosSetpoint : public Setpoint {
 public:
  using Setpoint::Setpoint;

  inline geometry_msgs::PoseStamped toPose() const {
    geometry_msgs::PoseStamped pose;
    pose.header.stamp = ros::Time(this->state.t);
    pose.header.frame_id = "world";
    pose.pose.position = toRosPoint(this->state.p);
    pose.pose.orientation = toRosQuaternion(this->state.q());
    return pose;
  }
};

}  // namespace agi