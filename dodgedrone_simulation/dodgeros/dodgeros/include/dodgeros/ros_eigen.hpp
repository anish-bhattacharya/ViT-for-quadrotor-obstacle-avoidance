#pragma once

#include "dodgelib/math/types.hpp"
#include "dodgelib/types/command.hpp"
#include "dodgelib/types/quad_state.hpp"
#include "dodgeros_msgs/Command.h"
#include "dodgeros_msgs/QuadState.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/Vector3.h"
#include "std_msgs/Bool.h"

namespace agi {

template<typename T>
inline geometry_msgs::Point toRosPoint(const T& vector) {
  geometry_msgs::Point point;
  point.x = vector.x();
  point.y = vector.y();
  point.z = vector.z();
  return point;
}

template<typename T>
inline geometry_msgs::Vector3 toRosVector(const T& vector) {
  geometry_msgs::Vector3 vec;
  vec.x = vector.x();
  vec.y = vector.y();
  vec.z = vector.z();
  return vec;
}

template<typename T>
inline geometry_msgs::Quaternion toRosQuaternion(const T& quaternion) {
  geometry_msgs::Quaternion ros_quaternion;
  ros_quaternion.w = quaternion.w();
  ros_quaternion.x = quaternion.x();
  ros_quaternion.y = quaternion.y();
  ros_quaternion.z = quaternion.z();
  return ros_quaternion;
}

inline dodgeros_msgs::Command toRosCommand(const agi::Command& command) {
  dodgeros_msgs::Command ros_command;
  ros_command.header.stamp = ros::Time(command.t);
  ros_command.t = command.t;
  ros_command.is_single_rotor_thrust = command.isSingleRotorThrusts();
  ros_command.collective_thrust = command.collective_thrust;
  ros_command.bodyrates.x = command.omega.x();
  ros_command.bodyrates.y = command.omega.y();
  ros_command.bodyrates.z = command.omega.z();

  for (int i = 0; i < 4; i++) {
    ros_command.thrusts[i] = command.thrusts[i];
  }

  return ros_command;
}

inline dodgeros_msgs::QuadState toRosQuadState(const agi::QuadState& state) {
  dodgeros_msgs::QuadState msg;
  msg.t = state.t;
  msg.header.frame_id = "world";
  msg.header.stamp =
    std::isfinite(state.t) ? ros::Time(state.t) : ros::Time::now();
  msg.pose.position = toRosPoint(state.p);
  msg.pose.orientation = toRosQuaternion(state.q());
  msg.velocity.linear = toRosVector(state.v);
  msg.velocity.angular = toRosVector(state.w);
  msg.acceleration.linear = toRosVector(state.a);
  msg.acceleration.angular = toRosVector(state.tau);
  msg.acc_bias = toRosVector(state.ba);
  msg.gyr_bias = toRosVector(state.bw);

  msg.jerk = toRosVector(state.j);
  msg.snap = toRosVector(state.s);

  msg.motors.push_back(state.mot[0]);
  msg.motors.push_back(state.mot[1]);
  msg.motors.push_back(state.mot[2]);
  msg.motors.push_back(state.mot[3]);

  return msg;
}

template<typename T>
inline agi::Command fromRosCommand(const T& ros_command) {
  if (ros_command.is_single_rotor_thrust) {
    return Command(ros_command.t,
                   Vector<4>(ros_command.thrusts[0], ros_command.thrusts[1],
                             ros_command.thrusts[2], ros_command.thrusts[3]));
  } else {
    return Command(ros_command.t, ros_command.collective_thrust,
                   Vector<3>(ros_command.bodyrates.x, ros_command.bodyrates.y,
                             ros_command.bodyrates.z));
  }
}

template<typename T>
inline agi::Command fromRosCommand(const T& ros_command, const Scalar& time) {
  if (ros_command.is_single_rotor_thrust) {
    return Command(time,
                   Vector<4>(ros_command.thrusts[0], ros_command.thrusts[1],
                             ros_command.thrusts[2], ros_command.thrusts[3]));
  } else {
    return Command(time, ros_command.collective_thrust,
                   Vector<3>(ros_command.bodyrates.x, ros_command.bodyrates.y,
                             ros_command.bodyrates.z));
  }
}


inline Quaternion fromRosQuaternion(
  const geometry_msgs::Quaternion quaternion) {
  return Quaternion(quaternion.w, quaternion.x, quaternion.y, quaternion.z);
}

template<typename T>
inline Vector<3> fromRosVec3(const T& point) {
  return Vector<3>(point.x, point.y, point.z);
}

template<typename T>
inline bool fromRosThrusts(const T& ros_thrusts, Vector<4>* const agi_thrusts) {
  if (ros_thrusts.size() != 4) {
    ROS_ERROR("Rotor thrusts.size: %lu", ros_thrusts.size());
    return false;
  }
  *agi_thrusts = Vector<4>(ros_thrusts.at(0), ros_thrusts.at(1),
                           ros_thrusts.at(2), ros_thrusts.at(3));
  return true;
}


}  // namespace agi