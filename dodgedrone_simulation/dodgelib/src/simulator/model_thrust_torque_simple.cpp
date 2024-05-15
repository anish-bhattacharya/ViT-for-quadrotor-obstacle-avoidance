#include "dodgelib/simulator/model_thrust_torque_simple.hpp"

#include <iostream>
namespace agi {

ModelThrustTorqueSimple::ModelThrustTorqueSimple(Quadrotor quad)
  : ModelBase(quad) {
  updateQuad(quad);
}

bool ModelThrustTorqueSimple::updateQuad(const Quadrotor& quad) {
  if (!quad.valid()) return false;
  quad_ = quad;
  B_allocation_ = quad.getAllocationMatrix();
  return true;
}


bool ModelThrustTorqueSimple::run(const Ref<const Vector<QS::SIZE>> state,
                                  Ref<Vector<QS::SIZE>> derivative) const {
  // Compute force and body torque in body frame
  const Vector<4> thrusts =
    quad_.motorOmegaToThrust(state.segment<QS::NMOT>(QS::MOT));
  const Vector<4> force_torques = B_allocation_ * thrusts;


  // Convert force and torque to acceleration and omega_dot
  const Vector<3> force(0.0, 0.0, force_torques[0]);
  const Matrix<3, 3> R = Quaternion(state(QS::ATTW), state(QS::ATTX),
                                    state(QS::ATTY), state(QS::ATTZ))
                           .toRotationMatrix();
  const Vector<3> omega(state(QS::OMEX), state(QS::OMEY), state(QS::OMEZ));


  derivative.segment<QS::NVEL>(QS::VEL) += R * force / quad_.m_ + GVEC;
  derivative.segment<QS::NOME>(QS::OME) +=
    quad_.J_inv_ * force_torques.segment<3>(1);

  return true;
}


}  // namespace agi
