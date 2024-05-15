#include "dodgelib/simulator/low_level_controller_base.hpp"

namespace agi {

LowLevelControllerBase::LowLevelControllerBase(const Quadrotor& quad,
                                               const std::string& name)
  : Module(name), quad_(quad) {
  state_.setZero();
  motor_omega_des_.setZero();
}


bool LowLevelControllerBase::setState(const QuadState& state) {
  if (!state.valid()) return false;
  state_ = state;
  return true;
}


bool LowLevelControllerBase::setCommand(const Command& cmd) {
  if (!cmd.valid()) return false;
  cmd_ = cmd;
  return true;
}


bool LowLevelControllerBase::getMotorCommand(Ref<Vector<4>> motor_omega) {
  run();
  if (!motor_omega_des_.allFinite()) return false;
  motor_omega = motor_omega_des_;
  return true;
}


bool LowLevelControllerBase::updateQuad(const Quadrotor& quad) {
  if (!quad.valid()) return false;
  quad_ = quad;
  return true;
}

bool LowLevelControllerBase::setParamDir(const fs::path& param_dir) {
  if (!fs::exists(param_dir)) {
    return false;
  }
  param_dir_ = param_dir;
  return true;
}

}  // namespace agi
