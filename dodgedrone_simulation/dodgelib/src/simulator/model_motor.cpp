#include "dodgelib/simulator/model_motor.hpp"

namespace agi {


bool ModelMotor::run(const Ref<const Vector<QS::SIZE>> state,
                     Ref<Vector<QS::SIZE>> derivative) const {
  // first order system model
  if (!std::isfinite(quad_.motor_tau_inv_)) {
    derivative.segment<QS::NMOT>(QS::MOT).setZero();
  } else {
    derivative.segment<QS::NMOT>(QS::MOT) =
      quad_.motor_tau_inv_ * (state.segment<QS::NMOTDES>(QS::MOTDES) -
                              state.segment<QS::NMOT>(QS::MOT));
  }

  return true;
}
}  // namespace agi
