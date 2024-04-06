#include "dodgelib/types/quad_state.hpp"

namespace agi {

QuadState::QuadState() {}

QuadState::QuadState(const Scalar t, const Vector<IDX::SIZE>& x) : x(x), t(t) {}

QuadState::QuadState(const Scalar t, const Vector<3>& position,
                     const Scalar yaw)
  : x(Vector<IDX::SIZE>::Zero()), t(t) {
  p = position;
  q(yaw);
}

QuadState::QuadState(const QuadState& state) : x(state.x), t(state.t) {}

QuadState::~QuadState() {}

Quaternion QuadState::q() const {
  return Quaternion(x(ATTW), x(ATTX), x(ATTY), x(ATTZ));
}

void QuadState::q(const Quaternion quaternion) {
  x(IDX::ATTW) = quaternion.w();
  x(IDX::ATTX) = quaternion.x();
  x(IDX::ATTY) = quaternion.y();
  x(IDX::ATTZ) = quaternion.z();
}


void QuadState::q(const Scalar angle, const Vector<3>& axis) {
  const Quaternion q_aa(Eigen::AngleAxis<Scalar>(angle, axis));
  x(IDX::ATTW) = q_aa.w();
  x(IDX::ATTX) = q_aa.x();
  x(IDX::ATTY) = q_aa.y();
  x(IDX::ATTZ) = q_aa.z();
}

Matrix<3, 3> QuadState::R() const {
  return Quaternion(x(ATTW), x(ATTX), x(ATTY), x(ATTZ)).toRotationMatrix();
}

void QuadState::setZero(const bool& reset_time) {
  if (reset_time) {
    t = 0.0;
  }
  x.setZero();
  x(ATTW) = 1.0;
}

void QuadState::linearize() { qx.normalize(); }

Scalar QuadState::getYaw(const Scalar yaw) const {
  const Vector<3> x_b = q() * Vector<3>::UnitX();
  const Vector<3> x_proj = Vector<3>(x_b.x(), x_b.y(), 0);
  if (x_proj.norm() < 1e-3) return yaw;
  const Vector<3> x_proj_norm = x_proj.normalized();
  const Vector<3> cross = Vector<3>::UnitX().cross(x_proj_norm);
  const Scalar angle = asin(cross.z());
  if (x_proj_norm.x() >= 0.0) return angle;
  if (x_proj_norm.y() >= 0.0) return M_PI - angle;
  return -M_PI - angle;
}

void QuadState::applyYaw(const Scalar angle) {
  const Scalar yaw = getYaw();
  const Scalar new_yaw = std::isfinite(yaw) ? yaw : 0.0;
  q(q() * Eigen::AngleAxis(new_yaw, Vector<3>::UnitZ()));
  if (qx(0) < 0.0) qx = -qx;
}

bool QuadState::operator==(const QuadState& rhs) const {
  return t == rhs.t && x.isApprox(rhs.x);
}

bool QuadState::isApprox(const QuadState& rhs, const Scalar tol) const {
  return std::abs(t - rhs.t) < tol && x.isApprox(rhs.x, tol);
}

QuadState QuadState::getHoverState() const {
  QuadState hover_state;
  hover_state.setZero();
  hover_state.t = t;
  hover_state.p = p;
  hover_state.q(getYaw(0.0));
  return hover_state;
}

std::ostream& operator<<(std::ostream& os, const QuadState& state) {
  os.precision(6);
  os << std::scientific;
  os << "State at " << state.t << "s:\n";
  os.precision(3);
  os << "p=  [" << state.p.transpose() << "]\n"
     << "q=  [" << state.qx.transpose() << "]\n"
     << "v=  [" << state.v.transpose() << "]\n"
     << "w=  [" << state.w.transpose() << "]\n"
     << "a=  [" << state.a.transpose() << "]\n"
     << "j=  [" << state.j.transpose() << "]\n"
     << "s=  [" << state.s.transpose() << "]\n"
     << "tau=[" << state.tau.transpose() << "]\n"
     << "bw= [" << state.bw.transpose() << "]\n"
     << "ba= [" << state.ba.transpose() << "]\n"
     << "mot=[" << state.mot.transpose() << "]\n"
     << "dmt=[" << state.motdes.transpose() << "]" << std::endl;
  os.precision();
  os.unsetf(std::ios::scientific);
  return os;
}

}  // namespace agi
