#pragma once

#include "dodgelib/reference/reference_base.hpp"

namespace agi {

class SampledTrajectory : public ReferenceBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SampledTrajectory(const SetpointVector& setpoints);
  virtual ~SampledTrajectory() = default;

  virtual Setpoint getSetpoint(const QuadState& state, const Scalar t) override;
  virtual Setpoint getStartSetpoint() override;
  virtual Setpoint getEndSetpoint() override;

 private:
  Setpoint interpolateSetpoints(const Setpoint& setpoint_1,
                                const Setpoint& setpoint_2,
                                const Scalar x) const;
  SetpointVector setpoints_;
};

}  // namespace agi
