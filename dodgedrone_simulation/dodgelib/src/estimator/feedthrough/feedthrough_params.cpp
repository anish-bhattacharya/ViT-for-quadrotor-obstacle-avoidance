#include "dodgelib/estimator/feedthrough/feedthrough_params.hpp"

namespace agi {

FeedthroughParameters::FeedthroughParameters()
  : transform_enabled_(false),
    roll_(0.0),
    pitch_(0.0),
    yaw_(0.0),
    pos_offset_(Vector<3>::Zero()) {}


bool FeedthroughParameters::load(const Yaml& node) {
  if (node.isNull()) return false;

  transform_enabled_ = node["transform_enabled"].as<bool>();
  roll_ = node["roll"].as<Scalar>() * M_PI / 180.0;
  pitch_ = node["pitch"].as<Scalar>() * M_PI / 180.0;
  yaw_ = node["yaw"].as<Scalar>() * M_PI / 180.0;
  node["pos_offset"] >> pos_offset_;

  return valid();
}

bool FeedthroughParameters::valid() const {
  bool check = true;
  check &= std::isfinite(transform_enabled_);
  check &= std::isfinite(roll_);
  check &= std::isfinite(pitch_);
  check &= std::isfinite(yaw_);
  check &= pos_offset_.allFinite();
  return check;
}

}  // namespace agi
