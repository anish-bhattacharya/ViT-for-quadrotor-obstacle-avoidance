#include "dodgelib/simulator/quadrotor_simulator.hpp"

#include "dodgelib/math/gravity.hpp"

namespace agi {

QuadrotorSimulator::QuadrotorSimulator(const Quadrotor &quad)
  : SimulatorBase("Quadrotor Simulator"),
    quadrotor_(quad),
    ctrl_(std::make_shared<LowLevelControllerSimple>(quad)) {
  updateQuad(quad);
  model_pipeline_.emplace_back(std::make_shared<ModelInit>(ModelInit{quad}));
  reset();
}

bool QuadrotorSimulator::run(const Command &cmd, const Scalar ctl_dt) {
  if (!setCommand(cmd)) return false;
  return run(ctl_dt);
}

bool QuadrotorSimulator::run(const Scalar ctl_dt) {
  if (!state_.valid()) return false;

  QuadState next_state = state_;
  const Scalar max_dt = integrator_ptr_->dtMax();
  Scalar remain_ctl_dt = ctl_dt;

  // simulation loop
  while (remain_ctl_dt > 0.0) {
    const Scalar sim_dt = std::min(remain_ctl_dt, max_dt);

    ctrl_->setState(state_);
    ctrl_->getMotorCommand(state_.motdes);

    if (!std::isfinite(quadrotor_.motor_tau_inv_)) {
      state_.mot = state_.motdes;
    }

    integrator_ptr_->step(state_.x, sim_dt, next_state.x);
    updateState(next_state, sim_dt);

    if (state_.p.z() <= 0.0) {
      state_.p.z() = 0.0;
      state_.v.z() = 0.0;
      state_.a.z() = 0.0;
    }

    remain_ctl_dt -= sim_dt;
  }

  state_.qx.normalize();

  state_.t += ctl_dt;
  return true;
}

bool QuadrotorSimulator::reset(const bool &reset_time) {
  state_.setZero(reset_time);
  return true;
}

bool QuadrotorSimulator::reset(const QuadState &state) {
  if (!state.valid()) return false;
  state_ = state;
  return true;
}


bool QuadrotorSimulator::computeDynamics(
  const Ref<const Vector<QS::SIZE>> state,
  Ref<Vector<QS::SIZE>> derivative) const {
  for (auto &m : model_pipeline_) {
    m->run(state, derivative);
  }
  return true;
}


void QuadrotorSimulator::updateState(const QuadState &next_state,
                                     Scalar sim_dt) {
  // Calculate mean acceleration and body torque over time step
  const Vector<3> delta_vel = next_state.v - state_.v;

  state_ = next_state;
  state_.a = delta_vel / sim_dt;
}

bool QuadrotorSimulator::setCommand(const Command &cmd) {
  return ctrl_->setCommand(cmd);
}

bool QuadrotorSimulator::setState(const QuadState &state) {
  if (!state.valid()) return false;
  state_ = state;
  return true;
}

bool QuadrotorSimulator::getState(QuadState *const state) const {
  if (!state_.valid()) return false;
  *state = state_;
  return true;
}

bool QuadrotorSimulator::getQuadrotor(Quadrotor *const quad) const {
  if (!quadrotor_.valid()) return false;
  *quad = quadrotor_;
  return true;
}

const Quadrotor &QuadrotorSimulator::getQuadrotor() const { return quadrotor_; }

const std::shared_ptr<LowLevelControllerBase>
QuadrotorSimulator::getLowLevelController() const {
  return ctrl_;
}

DynamicsFunction QuadrotorSimulator::getDynamics() const {
  return std::bind(
    static_cast<bool (QuadrotorSimulator::*)(const Ref<const Vector<QS::SIZE>>,
                                             Ref<Vector<QS::SIZE>>) const>(
      &QuadrotorSimulator::computeDynamics),
    this, std::placeholders::_1, std::placeholders::_2);
}

bool QuadrotorSimulator::updateQuad(const Quadrotor &quad) {
  if (!quad.valid()) {
    return false;
  }
  quadrotor_ = quad;
  ctrl_->updateQuad(quad);
  for (auto &m : model_pipeline_) m->updateQuad(quad);
  integrator_ptr_ = std::make_unique<IntegratorRK4>(this->getDynamics());
  return true;
}

bool QuadrotorSimulator::setLowLevelController(const std::string &llc_name) {
  if (llc_name == "Simple") {
    ctrl_ = std::make_shared<LowLevelControllerSimple>(quadrotor_);
    logger_.info("Enabled simple low level controller.");
    return true;
  }
  logger_.error("Unknown low level controller specified: [%s]",
                llc_name.c_str());
  return false;
}

}  // namespace agi
