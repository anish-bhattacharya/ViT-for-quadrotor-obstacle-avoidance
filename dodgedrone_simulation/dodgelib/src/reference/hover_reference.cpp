#include "dodgelib/reference/hover_reference.hpp"

#include "dodgelib/math/gravity.hpp"

namespace agi {

HoverReference::HoverReference(const QuadState& state, const Scalar duration)
  : ReferenceBase(state.getHoverState(), duration, "Hover Reference") {}


Setpoint HoverReference::getSetpoint(const QuadState& state, const Scalar t) {
  Setpoint setpoint;
  setpoint.state = start_state_;
  setpoint.state.t = t;
  setpoint.input = Command(t, G, Vector<3>::Zero());

  return setpoint;
}

}  // namespace agi
