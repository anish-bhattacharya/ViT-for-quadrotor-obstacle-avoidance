#pragma once

#include "dodgelib/reference/reference_base.hpp"

namespace agi {


class VelocityReference : public ReferenceBase {
 public:
  VelocityReference(const QuadState& state,
                    const bool update_from_estimate = false,
                    const Scalar timeout = 1.0);

  virtual Setpoint getSetpoint(const QuadState& state, const Scalar t) override;

  bool update(const Vector<3>& velocity, const Scalar yaw_rate);

  virtual bool isVelocityRefernce() const override { return true; }
  virtual bool isAbsolute() const override { return false; }

 private:
  void updateTo(const QuadState& state);

  const bool update_from_estimate_;
  Vector<3> v_ = Vector<3>::Zero();
  Scalar yaw_last_{0.0};
  Scalar yaw_rate_{0.0};
  const Scalar timeout_{1.0};
  Scalar t_last_update_{NAN};
  Logger logger_{name()};
};

}  // namespace agi
