#include <string>

#include "dodgelib/reference/reference_base.hpp"
#include "dodgelib/sampler/time_based/time_sampler.hpp"
#include "dodgeros/ros_setpoint.hpp"
#include "nav_msgs/Path.h"
#include "ros/ros.h"


namespace agi {
class RosTrajVisualizer {
 public:
  RosTrajVisualizer(const std::string name, const ros::NodeHandle nh,
                    const ros::NodeHandle pnh, const Scalar traj_viz_dt = 0.1,
                    const Scalar sphere_size = 0.1);
  RosTrajVisualizer(const std::string name)
    : RosTrajVisualizer(name, ros::NodeHandle(), ros::NodeHandle("~")) {}
  ~RosTrajVisualizer();
  bool visualize(const std::vector<std::shared_ptr<ReferenceBase>> references);
  bool visualize(const SetpointVector& setpoints, const std::string& name,
                 const int trajectory_index = 0);

 private:
  bool visualize(nav_msgs::Path& path, const SetpointVector& setpoints,
                 const std::string& name, const int trajectory_index);

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Publisher path_pub_;
  ros::Publisher marker_pub_;

  std::string name_;
  Scalar sphere_size_;

  std::shared_ptr<TimeSampler> viz_sampler_;

  std::vector<Vector<3>> colors_{
    Vector<3>(0.0, 0.0, 0.0), Vector<3>(0.0, 0.0, 1.0),
    Vector<3>(0.0, 1.0, 0.0), Vector<3>(0.0, 1.0, 1.0),
    Vector<3>(1.0, 0.0, 0.0), Vector<3>(1.0, 0.0, 1.0),
    Vector<3>(1.0, 1.0, 0.0), Vector<3>(1.0, 1.0, 1.0)};
};
}  // namespace agi