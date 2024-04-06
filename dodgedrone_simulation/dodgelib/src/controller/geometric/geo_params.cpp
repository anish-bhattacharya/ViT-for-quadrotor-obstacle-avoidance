#include "dodgelib/controller/geometric/geo_params.hpp"

namespace agi {

GeometricControllerParams::GeometricControllerParams()
  : kp_acc_(18.0, 18.0, 18.0),
    kd_acc_(8.0, 8.0, 8.0),
    kp_rate_(20.0, 20.0, 2.0),
    kp_att_xy_(150.0),
    kp_att_z_(2.0),
    filter_sampling_frequency_(100),
    filter_cutoff_frequency_(10),
    drag_compensation_(false) {}

bool GeometricControllerParams::load(const Yaml& node) {
  node["kpacc"] >> kp_acc_;
  node["kdacc"] >> kd_acc_;
  node["kprate"] >> kp_rate_;
  kp_att_z_ = node["kpatt_z"].as<Scalar>();
  kp_att_xy_ = node["kpatt_xy"].as<Scalar>();
  filter_sampling_frequency_ = node["filter_sampling_frequency"].as<Scalar>();
  filter_cutoff_frequency_ = node["filter_cutoff_frequency"].as<Scalar>();
  drag_compensation_ = node["drag_compensation"].as<bool>();
  node["p_err_max"].getIfDefined(p_err_max_);
  node["v_err_max"].getIfDefined(v_err_max_);

  return valid();
}

bool GeometricControllerParams::valid() const {
  bool check = true;
  check &= kp_acc_.allFinite() && (kp_acc_.array() >= 0.0).all();
  check &= kd_acc_.allFinite() && (kd_acc_.array() >= 0.0).all();
  check &= kp_rate_.allFinite() && (kp_rate_.array() >= 0.0).all();
  check &= (kp_att_z_ > 0.0);
  check &= (kp_att_xy_ >= kp_att_z_);
  check &= (filter_sampling_frequency_ > 2.0 * filter_cutoff_frequency_);
  check &= (filter_cutoff_frequency_ > 0.0);
  check &= p_err_max_.allFinite() && (p_err_max_.array() >= 0.0).all();
  check &= v_err_max_.allFinite() && (v_err_max_.array() >= 0.0).all();

  return check;
}

}  // namespace agi
