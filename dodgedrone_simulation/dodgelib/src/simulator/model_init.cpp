#include "dodgelib/simulator/model_init.hpp"

namespace agi {


bool ModelInit::run(const Ref<const Vector<QS::SIZE>> state,
                    Ref<Vector<QS::SIZE>> derivative) const {
  if (!state.segment<QS::DYN>(0).allFinite()) return false;

  // The derivative must be zeroed at first
  derivative.setZero();
  return true;
}


}  // namespace agi
