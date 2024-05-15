#include "dodgelib/types/quadrotor.hpp"

#include <cmath>
#include <iostream>

#include "dodgelib/math/math.hpp"

namespace agi {

Quadrotor::Quadrotor(const Scalar m, const Scalar l)
  : m_(m),
    t_BM_(
      l * sqrt(0.5) *
      (Matrix<3, 4>() << 1, -1, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0).finished()),
    J_(m_ / 12.0 * l * l *
       Vector<3>(2.25, 2.25, 4)
         .asDiagonal()),  // from cubic volume inertia with height=0.5 l
    J_inv_(J_.inverse()),
    motor_omega_min_(150.0),
    motor_omega_max_(2000.0),
    motor_tau_inv_(1.0 / 0.033),
    thrust_map_(1.562522e-06, 0.0, 0.0),
    torque_map_(1.908873e-08, 0.0, 0.0),
    kappa_(torque_map_(0) / thrust_map_(0)),
    thrust_min_(0.0),
    thrust_max_(motor_omega_max_ * motor_omega_max_ * thrust_map_(0) +
                motor_omega_max_ * thrust_map_(1) + thrust_map_(2)),
    rotors_config_(RotorConfig::cross),
    omega_max_(Vector<3>::Constant(6.0)),
    aero_coeff_1_(Vector<3>::Zero()),
    aero_coeff_3_(Vector<3>::Zero()),
    aero_coeff_h_(0.0) {}

Quadrotor::Quadrotor() : Quadrotor(1.0, 0.25) {}

bool Quadrotor::dynamics(const QuadState& state,
                         QuadState* const derivative) const {
  if (derivative == nullptr) return false;
  return dynamics(state.x, derivative->x);
}

bool Quadrotor::dynamics(const Ref<const Vector<QS::SIZE>> state,
                         Ref<Vector<QS::SIZE>> derivative) const {
  if (!state.segment<QS::DYN>(0).allFinite()) return false;

  derivative.setZero();
  const Vector<3> omega(state(QS::OMEX), state(QS::OMEY), state(QS::OMEZ));
  const Quaternion q_omega(0, omega.x(), omega.y(), omega.z());

  derivative.segment<QS::NPOS>(QS::POS) = state.segment<QS::NVEL>(QS::VEL);
  derivative.segment<QS::NATT>(QS::ATT) =
    0.5 * Q_right(q_omega) * state.segment<QS::NATT>(QS::ATT);
  derivative.segment<QS::NVEL>(QS::VEL) = state.segment<QS::NACC>(QS::ACC);
  derivative.segment<QS::NOME>(QS::OME) =
    J_inv_ * (state.segment<QS::NTAU>(QS::TAU) - omega.cross(J_ * omega));

  return true;
}

bool Quadrotor::jacobian(const Ref<const Vector<QS::SIZE>> state,
                         Ref<Matrix<QS::SIZE, QS::SIZE>> jac) const {
  if (!state.segment<QS::DYN>(0).allFinite()) return false;

  jac.setZero();

  const Quaternion q(state(QS::ATTW), state(QS::ATTX), state(QS::ATTY),
                     state(QS::ATTZ));
  const Vector<3> omega(state(QS::OMEX), state(QS::OMEY), state(QS::OMEZ));
  const Quaternion q_omega(0, omega.x(), omega.y(), omega.z());

  // Set dp/dv to identity.
  jac.block<QS::NPOS, QS::NVEL>(QS::POS, QS::VEL) =
    Matrix<QS::NPOS, QS::NVEL>::Identity();

  // Set dq/dq.
  jac.block<QS::NATT, QS::NATT>(QS::ATT, QS::ATT) = 0.5 * Q_right(q_omega);

  // Set dq/dw.
  jac.block<QS::NATT, QS::NOME>(QS::ATT, QS::OME) =
    0.5 * Q_left(q).block<QS::NATT, QS::NOME>(0, 1);

  // Set dv/da to identity.
  jac.block<QS::NVEL, QS::NACC>(QS::VEL, QS::ACC) =
    Matrix<QS::NVEL, QS::NACC>::Identity();

  // Set dw/dtau to J_inv
  jac.block<QS::NOME, QS::NTAU>(QS::OME, QS::TAU) = J_inv_;

  // Set dw/dw to J_inv
  jac.block<QS::NOME, QS::NOME>(QS::OME, QS::OME) =
    -J_inv_ * (skew(omega) * J_ + skew(J_ * omega).transpose());

  return true;
}

bool Quadrotor::jacobian(const Ref<const Vector<QS::SIZE>> state,
                         SparseMatrix& jac) const {
  Matrix<QS::SIZE, QS::SIZE> jac_dense;
  const bool ret = jacobian(state, jac_dense);
  jac = jac_dense.sparseView();
  return ret;
}

DynamicsFunction Quadrotor::getDynamicsFunction() const {
  return std::bind(
    static_cast<bool (Quadrotor::*)(const Ref<const Vector<QS::SIZE>>,
                                    Ref<Vector<QS::SIZE>>) const>(
      &Quadrotor::dynamics),
    this, std::placeholders::_1, std::placeholders::_2);
}

bool Quadrotor::load(const Yaml& node) {
  m_ = node["mass"].as<Scalar>();

  Vector<3> J_diag = J_.diagonal();
  node["inertia"].getIfDefined(J_diag);
  J_ = J_diag.asDiagonal();
  J_inv_ = J_.inverse();
  if (node["arm_length"].isDefined()) {
    // Sanity check
    if (node["rotors_config"].isDefined()) {
      if (node["rotors_config"].as<std::string>() == "plus") {
        throw ParameterException(
          "Parameter \"arm_length\" cannot be used with \"+\" configuration. "
          "Use \"t_BM\" or \"tbm_XX\" instead.");
      }
    }

    const Scalar l = node["arm_length"].as<Scalar>();
    t_BM_ =
      sqrt(0.5) * l *
      (Matrix<3, 4>() << 1, -1, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0).finished();

  } else if (node["t_BM"].isDefined()) {
    node["t_BM"] >> t_BM_;
  } else if (node["tbm_fr"].isDefined() && node["tbm_bl"].isDefined() &&
             node["tbm_br"].isDefined() && node["tbm_fl"].isDefined()) {
    t_BM_(0, 0) = node["tbm_fr"][0].as<Scalar>();
    t_BM_(1, 0) = node["tbm_fr"][1].as<Scalar>();
    t_BM_(2, 0) = node["tbm_fr"][2].as<Scalar>();

    t_BM_(0, 1) = node["tbm_bl"][0].as<Scalar>();
    t_BM_(1, 1) = node["tbm_bl"][1].as<Scalar>();
    t_BM_(2, 1) = node["tbm_bl"][2].as<Scalar>();

    t_BM_(0, 2) = node["tbm_br"][0].as<Scalar>();
    t_BM_(1, 2) = node["tbm_br"][1].as<Scalar>();
    t_BM_(2, 2) = node["tbm_br"][2].as<Scalar>();

    t_BM_(0, 3) = node["tbm_fl"][0].as<Scalar>();
    t_BM_(1, 3) = node["tbm_fl"][1].as<Scalar>();
    t_BM_(2, 3) = node["tbm_fl"][2].as<Scalar>();

  } else {
    return false;
  }

  motor_omega_min_ = node["motor_omega_min"].as<Scalar>();
  motor_omega_max_ = node["motor_omega_max"].as<Scalar>();
  motor_tau_inv_ = node["motor_tau"].as<Scalar>();
  motor_tau_inv_ = 1.0 / motor_tau_inv_;

  node["thrust_map"].getIfDefined(thrust_map_);
  node["torque_map"].getIfDefined(torque_map_);
  kappa_ = node["kappa"].as<Scalar>();
  thrust_min_ = node["thrust_min"].as<Scalar>();
  thrust_max_ = node["thrust_max"].as<Scalar>();

  if (node["rotors_config"].isDefined()) {
    if (node["rotors_config"].as<std::string>() == "cross") {
      rotors_config_ = RotorConfig::cross;
    } else if (node["rotors_config"].as<std::string>() == "plus") {
      rotors_config_ = RotorConfig::plus;
    }
  }

  node["omega_max"] >> omega_max_;

  node["aero_coeff_1"].getIfDefined(aero_coeff_1_);
  node["aero_coeff_3"].getIfDefined(aero_coeff_3_);
  node["aero_coeff_h"].getIfDefined(aero_coeff_h_);

  return valid();
}

bool Quadrotor::valid() const {
  bool check = true;

  // Hardcode a limit of 100 if someone puts the mass in grams.
  check &= m_ > 0.0;
  check &= m_ < 100.0;

  check &= t_BM_.allFinite();
  check &= J_.allFinite();
  check &= J_inv_.allFinite();

  check &= motor_omega_min_ >= 0.0;
  check &= (motor_omega_max_ > motor_omega_min_);

  check &= thrust_map_.allFinite();
  check &= torque_map_.allFinite();
  check &= kappa_ > 0.0;
  check &= thrust_min_ >= 0.0;
  check &= (thrust_max_ > thrust_min_);

  check &= (omega_max_.array() > 0).all();

  check &= (aero_coeff_1_.array() >= 0.0).all();
  check &= (aero_coeff_3_.array() >= 0.0).all();
  check &= aero_coeff_h_ >= 0.0;

  return check;
}

Vector<4> Quadrotor::clampMotorOmega(const Vector<4>& omega) const {
  return omega.cwiseMax(motor_omega_min_).cwiseMin(motor_omega_max_);
}


Vector<4> Quadrotor::clampThrust(const Vector<4> thrusts) const {
  return thrusts.cwiseMax(thrust_min_).cwiseMin(thrust_max_);
}


Scalar Quadrotor::clampThrust(const Scalar thrust) const {
  return std::clamp(thrust, thrust_min_, thrust_max_);
}


Scalar Quadrotor::clampCollectiveThrust(const Scalar thrust) const {
  return std::clamp(thrust, collective_thrust_min(), collective_thrust_max());
}


Vector<3> Quadrotor::clampBodyrates(const Vector<3>& omega) const {
  return omega.cwiseMax(-omega_max_).cwiseMin(omega_max_);
}


Vector<4> Quadrotor::motorOmegaToThrust(const Vector<4>& omega) const {
  const Matrix<4, 3> omega_poly =
    (Matrix<4, 3>() << omega.cwiseProduct(omega), omega, Vector<4>::Ones())
      .finished();
  return omega_poly * thrust_map_;
}

Vector<4> Quadrotor::motorOmegaToTorque(const Vector<4>& omega) const {
  const Matrix<4, 3> omega_poly =
    (Matrix<4, 3>() << omega.cwiseProduct(omega), omega, Vector<4>::Ones())
      .finished();
  return omega_poly * torque_map_;
}

Vector<4> Quadrotor::motorThrustToOmega(const Vector<4>& thrusts) const {
  // midnight formula
  const Scalar scale = 1.0 / (2.0 * thrust_map_(0));
  const Scalar offset = -thrust_map_(1) * scale;

  const Array<4, 1> root =
    (std::pow(thrust_map_(1), 2) -
     4.0 * thrust_map_(0) * (thrust_map_(2) - thrusts.array()))
      .sqrt();

  return offset + scale * root;
}


Matrix<4, 4> Quadrotor::getAllocationMatrix() const {
  // compute column-wise cross product
  // tau_i = t_BM_i x F_i
  return (Matrix<4, 4>() << Vector<4>::Ones().transpose(), t_BM_.row(1),
          -t_BM_.row(0), kappa_ * Vector<4>(-1, -1, 1, 1).transpose())
    .finished();
}

std::ostream& operator<<(std::ostream& os, const Quadrotor& quad) {
  os.precision(3);
  os << std::scientific;
  os << "Quadrotor:\n"
     << "mass =             [" << quad.m_ << "]\n"
     << "t_BM =             [" << quad.t_BM_.row(0) << "]\n"
     << "                   [" << quad.t_BM_.row(1) << "]\n"
     << "                   [" << quad.t_BM_.row(2) << "]\n"
     << "inertia =          [" << quad.J_.row(0) << "]\n"
     << "                   [" << quad.J_.row(1) << "]\n"
     << "                   [" << quad.J_.row(2) << "]\n"
     << "motor_omega_min =  [" << quad.motor_omega_min_ << "]\n"
     << "motor_omega_max =  [" << quad.motor_omega_max_ << "]\n"
     << "motor_tau_inv =    [" << quad.motor_tau_inv_ << "]\n"
     << "thrust_map =       [" << quad.thrust_map_.transpose() << "]\n"
     << "torque_map =       [" << quad.torque_map_.transpose() << "]\n"
     << "kappa =            [" << quad.kappa_ << "]\n"
     << "thrust_min =       [" << quad.thrust_min_ << "]\n"
     << "thrust_max =       [" << quad.thrust_max_ << "]\n"
     << "omega_max =        [" << quad.omega_max_.transpose() << "]\n"
     << "aero_coeff_1 =     [" << quad.aero_coeff_1_.transpose() << "]\n"
     << "aero_coeff_3 =     [" << quad.aero_coeff_3_.transpose() << "]"
     << std::endl;
  os.precision();
  os.unsetf(std::ios::scientific);
  return os;
}

}  // namespace agi
