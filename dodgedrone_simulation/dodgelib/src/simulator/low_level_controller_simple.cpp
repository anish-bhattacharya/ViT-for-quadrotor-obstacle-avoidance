#include "dodgelib/simulator/low_level_controller_simple.hpp"


namespace agi {

LowLevelControllerSimple::LowLevelControllerSimple(Quadrotor quad)
  : LowLevelControllerBase(quad) {
  updateQuad(quad);
}

bool LowLevelControllerSimple::updateQuad(const Quadrotor& quad) {
  quad_ = quad;
  B_allocation_ = quad.getAllocationMatrix();
  const Matrix<> B_allocation_dyn = B_allocation_;
  B_allocation_inv_ = B_allocation_dyn.inverse();
  return true;
}

bool LowLevelControllerSimple::setCommand(const Command& cmd) {
  if (!cmd.valid()) return false;
  cmd_ = cmd;

  if (cmd_.isRatesThrust()) {
    cmd_.collective_thrust =
      quad_.clampCollectiveThrust(cmd_.collective_thrust);
    cmd_.omega = quad_.clampBodyrates(cmd_.omega);
  }

  if (cmd_.isSingleRotorThrusts())
    cmd_.thrusts = quad_.clampThrust(cmd_.thrusts);

  return true;
}


void LowLevelControllerSimple::run() {
  // If command is invalid, just switch of motors
  if (!cmd_.valid()) {
    motor_omega_des_.setZero();
    return;
  }

  Vector<4> motor_thrusts;
  if (!cmd_.isSingleRotorThrusts()) {
    const Scalar force = quad_.m_ * cmd_.collective_thrust;
    const Vector<3> omega_err = cmd_.omega - state_.w;
    const Vector<3> body_torque_des = quad_.J_ * Kinv_ang_vel_tau_ * omega_err +
                                      state_.w.cross(quad_.J_ * state_.w);
    const Vector<4> thrust_torque(force, body_torque_des.x(),
                                  body_torque_des.y(), body_torque_des.z());

    motor_thrusts = B_allocation_inv_ * thrust_torque;
  } else {
    motor_thrusts = cmd_.thrusts;
  }

  motor_thrusts = quad_.clampThrust(motor_thrusts);
  motor_omega_des_ = quad_.motorThrustToOmega(motor_thrusts);
}


}  // namespace agi
