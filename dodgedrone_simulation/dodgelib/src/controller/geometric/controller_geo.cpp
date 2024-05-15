#include "dodgelib/controller/geometric/controller_geo.hpp"

#include "dodgelib/math/math.hpp"

namespace agi {

GeometricController::GeometricController(
  const Quadrotor& quad,
  const std::shared_ptr<GeometricControllerParams>& params)
  : ControllerBase("GEO"),
    quad_(quad),
    params_(params),
    filterAcc_(params->filter_cutoff_frequency_,
               params->filter_sampling_frequency_, 0.0),
    filterMot_(params->filter_cutoff_frequency_,
               params->filter_sampling_frequency_, 0.0) {}

GeometricController::~GeometricController() {}

bool GeometricController::getCommand(const QuadState& state,
                                     const SetpointVector& references,
                                     SetpointVector* const setpoints) {
  if (setpoints == nullptr) return false;
  setpoints->clear();

  if (!state.valid() || references.empty() ||
      !references.front().input.valid()) {
    logger_.error("Control inputs are not valid!");
    logger_.error("State is valid: [%d]!", state.valid());
    logger_.error("Setpoints are empty: [%d]!", references.empty());
    logger_.error("Setpoint is valid: [%d]!", references.front().input.valid());
    logger_ << references.front().input;
    return false;
  }

  Setpoint setpoint = references.front();

  // update filters
  const Vector<3> accB =
    imu_.valid() ? imu_.acc + state.ba : state.q().inverse() * (state.a - GVEC);

  filterMot_.add(state.mot);
  const Vector<4> wf = filterMot_();
  const Scalar thrust_f = quad_.thrust_map_(0) * wf.transpose() * wf;

  filterAcc_.add(accB);
  const Vector<3> acc_f = filterAcc_();

  // acc command
  Vector<3> acc_cmd;
  {
    Vector<3> pos_error = clip(setpoint.state.p - state.p, params_->p_err_max_);
    Vector<3> vel_error = clip(setpoint.state.v - state.v, params_->v_err_max_);

    acc_cmd = params_->kp_acc_.cwiseProduct(pos_error) +
              params_->kd_acc_.cwiseProduct(vel_error) + setpoint.state.a -
              GVEC;

    if (params_->drag_compensation_ && state.v.norm() > 3.0) {
      const Vector<3> acc_aero =
        state.q() * (thrust_f * Vector<3>::UnitZ() / quad_.m_ - acc_f);
      acc_cmd += acc_aero;
    }
  }
  const Scalar thrust_cmd = acc_cmd.norm() * quad_.m_;

  // attitude command
  Quaternion q_cmd;
  {
    const Quaternion q_c(Quaternion(
      Eigen::AngleAxis<Scalar>(setpoint.state.getYaw(), Vector<3>::UnitZ())));
    const Vector<3> y_c = q_c * Vector<3>::UnitY();
    const Vector<3> z_B = acc_cmd.normalized();
    const Vector<3> x_B = (y_c.cross(z_B)).normalized();
    const Vector<3> y_B = (z_B.cross(x_B)).normalized();
    const Matrix<3, 3> R_W_B((Matrix<3, 3>() << x_B, y_B, z_B).finished());
    const Quaternion q_des(R_W_B);

    q_cmd = q_des;
  }

  // angular acceleration command
  Vector<3> alpha_cmd;
  Vector<3> omega_cmd;
  {
    omega_cmd = tiltPrioritizedControl(state.q(), q_cmd);

    // angular rate / acceleration reference from Mellinger 2011
    const Vector<3> bx = state.q() * Vector<3>::UnitX();
    const Vector<3> by = state.q() * Vector<3>::UnitY();
    const Vector<3> bz = state.q() * Vector<3>::UnitZ();
    Vector<3> hw =
      quad_.m_ * (setpoint.state.j - bz.dot(setpoint.state.j) * bz);
    if (thrust_f >= 0.01) hw /= thrust_f;
    const Vector<3> w_ref =
      Vector<3>(-hw.dot(by), hw.dot(bx), setpoint.state.w(2));

    alpha_cmd = omega_cmd + params_->kp_rate_.cwiseProduct(w_ref - state.w);
  }

  QuadState state_cmd = state;
  state_cmd.tau = alpha_cmd;
  Command command;
  command.t = state.t;
  command.omega = omega_cmd;
  command.collective_thrust = thrust_cmd / quad_.m_;
  setpoints->push_back({state_cmd, command});

  return true;
}

Vector<3> GeometricController::tiltPrioritizedControl(const Quaternion& q,
                                                      const Quaternion& q_des) {
  // Attitude control method from Fohn 2020.
  const Quaternion q_e = q.inverse() * q_des;

  Matrix<3, 3> T_att = (Matrix<3, 3>() << params_->kp_att_xy_, 0.0, 0.0, 0.0,
                        params_->kp_att_xy_, 0.0, 0.0, 0.0, params_->kp_att_z_)
                         .finished();
  Vector<3> tmp = Vector<3>(q_e.w() * q_e.x() - q_e.y() * q_e.z(),
                            q_e.w() * q_e.y() + q_e.x() * q_e.z(), q_e.z());
  if (q_e.w() <= 0) tmp(2) *= -1.0;
  const Vector<3> rate_cmd =
    2.0 / std::sqrt(q_e.w() * q_e.w() + q_e.z() * q_e.z()) * T_att * tmp;

  return rate_cmd;
}


bool GeometricController::addImu(const ImuSample& imu) {
  imu_ = imu;
  return true;
}

}  // namespace agi
