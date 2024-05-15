#include "dodgeros/ros_traj_visualizer.hpp"

#include <ros/console.h>

#include <cmath>

#include "dodgeros/ros_eigen.hpp"
#include "nav_msgs/Path.h"
#include "visualization_msgs/MarkerArray.h"

namespace agi {
RosTrajVisualizer::RosTrajVisualizer(const std::string name,
                                     const ros::NodeHandle nh,
                                     const ros::NodeHandle pnh,
                                     const Scalar traj_viz_dt,
                                     const Scalar sphere_size)
  : nh_(nh), pnh_(pnh), name_(name), sphere_size_(sphere_size) {
  viz_sampler_ = std::make_shared<TimeSampler>(-1, traj_viz_dt);
  // Publishers
  path_pub_ = nh_.advertise<nav_msgs::Path>(name + "/path", 1);
  marker_pub_ = nh_.advertise<visualization_msgs::Marker>(name + "/markers", 1);
}

RosTrajVisualizer::~RosTrajVisualizer() {}

bool RosTrajVisualizer::visualize(
  const std::vector<std::shared_ptr<ReferenceBase>> references) {
  nav_msgs::Path path;
  path.header.stamp = ros::Time::now();
  path.header.frame_id = "world";

  int i = 1;
  for (const std::shared_ptr<ReferenceBase>& reference : references) {
    SetpointVector setpoints;
    viz_sampler_->getFull(reference, &setpoints);
    if (!visualize(path, setpoints, reference->name(), i)) return false;
    ++i;
  }

  path_pub_.publish(path);

  return true;
}

bool RosTrajVisualizer::visualize(const SetpointVector& setpoints,
                                  const std::string& name,
                                  const int trajectory_index) {
  nav_msgs::Path path;
  path.header.stamp = ros::Time::now();
  path.header.frame_id = "world";

  visualize(path, setpoints, name, trajectory_index);

  path_pub_.publish(path);

  return true;
}

bool RosTrajVisualizer::visualize(nav_msgs::Path& path,
                                  const SetpointVector& setpoints,
                                  const std::string& name,
                                  const int trajectory_index) {
  const ros::Time time_now = ros::Time::now();

  visualization_msgs::Marker marker;
  marker.header.stamp = time_now;
  marker.header.frame_id = "world";
  marker.ns = name;
  marker.id = trajectory_index;
  marker.lifetime = ros::Duration(1.0);
  marker.type = visualization_msgs::Marker::SPHERE_LIST;
  marker.pose.position = toRosPoint(Vector<3>::Zero());
  marker.pose.orientation = toRosQuaternion(Quaternion(1, 0, 0, 0));
  marker.scale.x = sphere_size_;
  marker.scale.y = sphere_size_;
  marker.scale.z = sphere_size_;
  marker.color.a = 1.0;
  marker.color.r = colors_.at(trajectory_index % (colors_.size() - 1) + 1).x();
  marker.color.g = colors_.at(trajectory_index % (colors_.size() - 1) + 1).y();
  marker.color.b = colors_.at(trajectory_index % (colors_.size() - 1) + 1).z();

  for (const Setpoint& setpoint : setpoints) {
    if (!std::isfinite(setpoint.state.t) || setpoint.state.t < 0.0 ||
        (setpoint.state.t > std::numeric_limits<uint32_t>::max()))
      continue;
    path.poses.push_back(static_cast<const RosSetpoint*>(&setpoint)->toPose());
    marker.points.push_back(toRosPoint(setpoint.state.p));
  }

  marker_pub_.publish(marker);
  return true;
}

}  // namespace agi
