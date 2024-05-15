#pragma once

#include "dodgelib/reference/reference_base.hpp"

namespace agi {


class HoverReference : public ReferenceBase {
 public:
  HoverReference(const QuadState& state, const Scalar duration = INF);

  virtual Setpoint getSetpoint(const QuadState& state, const Scalar t) override;

  virtual bool isHover() const override { return true; }
};

}  // namespace agi
