#include "dodgelib/simulator/model_rigid_body.hpp"

namespace agi {

bool ModelRigidBody::run(const Ref<const Vector<QS::SIZE>> state,
                         Ref<Vector<QS::SIZE>> derivative) const {
  const Vector<3> omega(state(QS::OMEX), state(QS::OMEY), state(QS::OMEZ));
  const Quaternion q_omega(0, omega.x(), omega.y(), omega.z());

  // Set up the derivative
  derivative.segment<QS::NPOS>(QS::POS) = state.segment<QS::NVEL>(QS::VEL);
  derivative.segment<QS::NATT>(QS::ATT) =
    0.5 * Q_right(q_omega) * state.segment<QS::NATT>(QS::ATT);
  derivative.segment<QS::NOME>(QS::OME) +=
    -quad_.J_inv_ * omega.cross(quad_.J_ * omega);

  return true;
}


}  // namespace agi
