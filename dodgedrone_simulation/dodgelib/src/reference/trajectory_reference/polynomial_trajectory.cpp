#include "dodgelib/reference/trajectory_reference/polynomial_trajectory.hpp"

#include "dodgelib/math/gravity.hpp"


namespace agi {

template<class PolyType>
PolynomialTrajectory<PolyType>::PolynomialTrajectory(
  const QuadState& start_state, const QuadState& end_state,
  const Vector<>& weights, const int order, const int continuity,
  const std::string& name)
  : ReferenceBase(start_state, end_state.t - start_state.t, name),
    end_state_(end_state),
    x_(order, weights, continuity),
    y_(order, weights, continuity),
    z_(order, weights, continuity),
    yaw_(5, Vector<3>(0, 0, 1)),
    states_({start_state, end_state}) {
  yaw_last_ = start_state_.getYaw();

  x_.scale(start_state_.t, duration_);
  y_.scale(start_state_.t, duration_);
  z_.scale(start_state_.t, duration_);
  yaw_.scale(start_state_.t, duration_);

  addStateConstraint(start_state_);
  addStateConstraint(end_state_);

  x_.solve();
  y_.solve();
  z_.solve();
  yaw_.solve();
}

template<>
PolynomialTrajectory<ClosedFormMinJerkAxis>::PolynomialTrajectory(
  const QuadState& start_state, const QuadState& end_state,
  const Vector<>& weights, const int order, const int continuity,
  const std::string& name)
  : ReferenceBase(start_state, end_state.t - start_state.t, name),
    end_state_(end_state) {
  yaw_last_ = start_state_.getYaw();

  x_.scale(start_state_.t, duration_);
  y_.scale(start_state_.t, duration_);
  z_.scale(start_state_.t, duration_);
  yaw_.scale(start_state_.t, duration_);

  addStateConstraint(start_state_);
  addStateConstraint(end_state_);

  x_.solve();
  y_.solve();
  z_.solve();
  yaw_.solve();
}

template<class PolyType>
PolynomialTrajectory<PolyType>::PolynomialTrajectory(
  const std::vector<QuadState>& states, const Vector<>& weights,
  const int order, const std::string& name)
  : ReferenceBase(states.front(), states.back().t - states.front().t, name),
    end_state_(states.back()),
    x_(order, weights, -1),
    y_(order, weights, -1),
    z_(order, weights, -1),
    yaw_(5, Vector<3>(0, 0, 1)),
    states_(states) {
  yaw_last_ = start_state_.getYaw();

  x_.scale(start_state_.t, duration_);
  y_.scale(start_state_.t, duration_);
  z_.scale(start_state_.t, duration_);
  yaw_.scale(start_state_.t, duration_);

  for (const QuadState& state : states_) addStateConstraint(state);

  if (!x_.solve()) std::cout << "Could not solve x-axis!" << std::endl;
  if (!y_.solve()) std::cout << "Could not solve y-axis!" << std::endl;
  if (!z_.solve()) std::cout << "Could not solve z-axis!" << std::endl;
  if (!yaw_.solve()) std::cout << "Could not solve yaw!" << std::endl;

  start_state_ = getState(start_state_.t);
  if (!start_state_.valid())
    std::cout << "Could not fill start state!" << std::endl;

  end_state_ = getState(end_state_.t);
  if (!end_state_.valid())
    std::cout << "Could not fill end state!" << std::endl;
}

template<class PolyType>
bool PolynomialTrajectory<PolyType>::addStateConstraint(
  const QuadState& state) {
  if (!std::isfinite(state.t)) return false;
  Matrix<> constraints = Matrix<>::Constant(3, 5, NAN);

  constraints.col(0) = state.p;
  constraints.col(1) = state.v;
  constraints.col(2) = state.a;
  constraints.col(3) = state.j;
  constraints.col(4) = state.s;

  x_.addConstraint(state.t, constraints.row(0).transpose());
  y_.addConstraint(state.t, constraints.row(1).transpose());
  z_.addConstraint(state.t, constraints.row(2).transpose());

  // ToDo: Yaw relative to initial yaw.
  const Scalar yaw_angle = state.getYaw();
  if (std::isfinite(yaw_angle))
    yaw_.addConstraint(state.t, Vector<3>(yaw_angle, state.w.z(), 0));

  return true;
}

template<class PolyType>
Setpoint PolynomialTrajectory<PolyType>::getSetpoint(const QuadState& state,
                                                     const Scalar time) {
  const Scalar t = std::isfinite(time) ? std::max(state.t, time) : state.t;
  Setpoint setpoint;
  if (!std::isfinite(t)) return setpoint;

  setpoint.state = getState(t);
  setpoint.input.t = time;
  setpoint.input.collective_thrust = (setpoint.state.a - GVEC).norm();
  setpoint.input.omega = setpoint.state.w;

  return setpoint;
}

template<class PolyType>
QuadState PolynomialTrajectory<PolyType>::getState(const Scalar time) const {
  QuadState state;
  if (!valid()) return state;

  const Scalar t = std::clamp(time, start_state_.t, end_state_.t);

  Matrix<> x = Matrix<>::Zero(5, 3);
  Vector<> yaw = Vector<>::Zero(3);

  x_.eval(t, x.col(0));
  y_.eval(t, x.col(1));
  z_.eval(t, x.col(2));

  state.setZero();
  state.t = t;
  state.p = x.row(0).transpose();
  state.v = x.row(1).transpose();
  state.a = x.row(2).transpose();
  state.j = x.row(3).transpose();
  state.s = x.row(4).transpose();

  const Vector<3> thrust_vec = state.a - GVEC;
  const Scalar thrust = thrust_vec.norm();
  const Quaternion q_pitch_roll =
    thrust > 1e-3 ? Quaternion::FromTwoVectors(Vector<3>::UnitZ(), thrust_vec)
                  : q_pitch_roll_last_;
  q_pitch_roll_last_ = q_pitch_roll;

  if (forward_heading_) {
    const Vector<3> v_body = q_pitch_roll.inverse() * state.v;
    if ((v_body.x() * v_body.x() + v_body.y() * v_body.y()) < 1e-6) {
      yaw(0) = std::atan2(v_body.y(), v_body.x());
    } else {
      yaw(0) = yaw_last_;
    }
  } else {
    yaw_.eval(t, yaw);
  }

  yaw_last_ = yaw(0);

  const Quaternion q_heading =
    Quaternion(Eigen::AngleAxis<Scalar>(yaw(0), Vector<3>::UnitZ()));
  const Quaternion q_att = q_pitch_roll * q_heading;
  state.q(q_att.normalized());

  // compute bodyrates
  const Vector<3> body_jerk = q_att.inverse() * state.j;
  state.w = Vector<3>(-1.0 / thrust * body_jerk[1], 1.0 / thrust * body_jerk[0],
                      yaw(1));

  return state;
}

template<class PolyType>
Setpoint PolynomialTrajectory<PolyType>::getStartSetpoint() {
  return getSetpoint(start_state_);
}

template<class PolyType>
Setpoint PolynomialTrajectory<PolyType>::getEndSetpoint() {
  return getSetpoint(end_state_);
}

template<class PolyType>
bool PolynomialTrajectory<PolyType>::solved() const {
  if (!x_.solved()) std::cout << "Could not solve x-axis!" << std::endl;
  if (!y_.solved()) std::cout << "Could not solve y-axis!" << std::endl;
  if (!z_.solved()) std::cout << "Could not solve z-axis!" << std::endl;
  if (!yaw_.solved()) std::cout << "Could not solve yaw!" << std::endl;
  return x_.solved() && y_.solved() && z_.solved() && yaw_.solved();
}

template<class PolyType>
bool PolynomialTrajectory<PolyType>::valid() const {
  return x_.solved() && y_.solved() && z_.solved() && yaw_.solved();
}

template<class PolyType>
Vector<> PolynomialTrajectory<PolyType>::evalTranslation(
  const Scalar time, const int order) const {
  return Vector<3>(x_(time, order), y_(time, order), z_(time, order));
}

template<class PolyType>
void PolynomialTrajectory<PolyType>::scale(const Scalar start_time,
                                           const Scalar duration) {
  if (std::isfinite(start_time)) {
    start_state_.t = start_time;
    end_state_.t = start_time + duration_;
  }

  if (std::isfinite(duration)) {
    duration_ = duration;
    end_state_.t = start_state_.t + duration_;
  }

  x_.scale(start_state_.t, duration_);
  y_.scale(start_state_.t, duration_);
  z_.scale(start_state_.t, duration_);
  yaw_.scale(start_state_.t, duration_);
}

template<class PolyType>
Scalar PolynomialTrajectory<PolyType>::scaleToLimits(const Quadrotor& quad,
                                                     const int iterations,
                                                     const Scalar tolerance) {
  return scaleToLimits(quad.collective_thrust_max() / quad.m_, iterations,
                       tolerance);
}

template<class PolyType>
Scalar PolynomialTrajectory<PolyType>::scaleToLimits(const Scalar acc_limit,
                                                     const int iterations,
                                                     const Scalar tolerance) {
  if (!std::isfinite(acc_limit) || acc_limit <= 0.0) return duration_;

  for (int k = 0; k < iterations; ++k) {
    const Scalar t_max_acc = findTimeMaxAcc();
    const Scalar max_acc = (evalTranslation(t_max_acc, 2) - GVEC).norm();
    const Scalar acc_ratio = max_acc / acc_limit;
    const Scalar acc_scale = std::pow(acc_ratio, 0.5);

    // Catch zero acceleration case.
    if (max_acc < 1e-3) break;

    duration_ *= acc_scale;
    end_state_.t = start_state_.t + duration_;
    x_.scale(NAN, duration_);
    y_.scale(NAN, duration_);
    z_.scale(NAN, duration_);
    yaw_.scale(NAN, duration_);
    if (std::abs(acc_scale - 1.0) < tolerance) break;
  }

  return duration_;
}

template<class PolyType>
Scalar PolynomialTrajectory<PolyType>::findTimeMaxAcc(
  const Scalar precision) const {
  return findTimeMaxAcc(NAN, NAN, precision);
}

template<class PolyType>
Scalar PolynomialTrajectory<PolyType>::findTimeMaxAcc(
  const Scalar t_start, const Scalar t_end, const Scalar precision) const {
  return findTime(
    getDuration() / 100, precision, t_start, t_end,
    [&](const Scalar t) { return (evalTranslation(t, 2) - GVEC).norm(); },
    [](const Scalar x, const Scalar x_max) { return x > x_max; });
}

template<class PolyType>
Scalar PolynomialTrajectory<PolyType>::findTimeMaxOmega(
  const Scalar precision) const {
  return findTimeMaxOmega(NAN, NAN, precision);
}

template<class PolyType>
Scalar PolynomialTrajectory<PolyType>::findTimeMaxOmega(
  const Scalar t_start, const Scalar t_end, const Scalar precision) const {
  return findTime(
    getDuration() / 100, precision, t_start, t_end,
    [this](const Scalar t) {
      const QuadState state = this->getState(t);
      return std::max(std::abs(state.w.x()), std::abs(state.w.y()));
    },
    [](const Scalar x, const Scalar x_max) { return x > x_max; });
}

template<class PolyType>
void PolynomialTrajectory<PolyType>::setForwardHeading(const bool forward) {
  forward_heading_ = forward;
}

template<class PolyType>
template<typename EvalFunc, typename CompFunc>
Scalar PolynomialTrajectory<PolyType>::findTime(
  const Scalar dt, const Scalar dt_min, const Scalar t_start,
  const Scalar t_end, EvalFunc eval, CompFunc comp) const {
  const Scalar start = std::isfinite(t_start) ? t_start : getStartTime();
  const Scalar end = std::isfinite(t_end) ? t_end : getEndTime();
  Scalar t_max = start;
  Scalar x_max = eval(t_max);

  // Sample between bounds [t_start, t_end].
  for (Scalar t = t_max + dt; t <= end; t += dt) {
    const Scalar x = eval(t);
    if (comp(x, x_max)) {
      x_max = x;
      t_max = t;
    }
  }

  // Recursion: search deeper until dt is less or equal dt_min.
  return dt <= dt_min ? t_max
                      : findTime(0.1 * dt, dt_min, std::max(t_max - dt, start),
                                 std::min(t_max + dt, end), eval, comp);
}

template class PolynomialTrajectory<Polynomial<>>;
template class PolynomialTrajectory<ClosedFormMinJerkAxis>;

}  // namespace agi
