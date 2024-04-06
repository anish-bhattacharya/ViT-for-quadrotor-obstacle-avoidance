#pragma once

#include "dodgelib/base/parameter_base.hpp"

namespace agi {

struct GeometricControllerParams : public ParameterBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GeometricControllerParams();

  using ParameterBase::load;
  virtual bool load(const Yaml& node) override;

  Vector<3> kp_acc_;
  Vector<3> kd_acc_;
  Vector<3> kp_rate_;
  Scalar kp_att_xy_;
  Scalar kp_att_z_;
  Scalar filter_sampling_frequency_;
  Scalar filter_cutoff_frequency_;
  bool drag_compensation_;
  Vector<3> p_err_max_ =
    Vector<3>::Constant(std::numeric_limits<Scalar>::max());
  Vector<3> v_err_max_ =
    Vector<3>::Constant(std::numeric_limits<Scalar>::max());
  bool valid() const override;
};

}  // namespace agi
