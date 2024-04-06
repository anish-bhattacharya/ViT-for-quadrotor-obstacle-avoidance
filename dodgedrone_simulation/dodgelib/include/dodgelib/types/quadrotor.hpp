#pragma once

#include <functional>

#include "dodgelib/base/parameter_base.hpp"
#include "dodgelib/math/types.hpp"
#include "dodgelib/types/quad_state.hpp"

namespace agi {

struct Quadrotor : public ParameterBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum class RotorConfig { plus, cross };

  Quadrotor(const Scalar m, const Scalar l);
  Quadrotor();

  bool dynamics(const QuadState& state, QuadState* const derivative) const;

  bool dynamics(const Ref<const Vector<QuadState::SIZE>> state,
                Ref<Vector<QuadState::SIZE>> derivative) const;

  bool jacobian(const Ref<const Vector<QuadState::SIZE>> state,
                Ref<Matrix<QuadState::SIZE, QuadState::SIZE>> jac) const;

  bool jacobian(const Ref<const Vector<QuadState::SIZE>> state,
                SparseMatrix& jac) const;

  DynamicsFunction getDynamicsFunction() const;

  using ParameterBase::load;
  bool load(const Yaml& node) override;
  bool valid() const override;

  // Helpers to apply limits.
  Vector<4> clampThrust(const Vector<4> thrusts) const;
  Scalar clampThrust(const Scalar thrust) const;
  Scalar clampCollectiveThrust(const Scalar thrust) const;
  Vector<4> clampMotorOmega(const Vector<4>& omega) const;
  Vector<3> clampBodyrates(const Vector<3>& omega) const;

  inline Scalar collective_thrust_min() const { return 4.0 * thrust_min_ / m_; }
  inline Scalar collective_thrust_max() const { return 4.0 * thrust_max_ / m_; }

  // Helpers for conversion
  Vector<4> motorOmegaToThrust(const Vector<4>& omega) const;
  Vector<4> motorOmegaToTorque(const Vector<4>& omega) const;
  Vector<4> motorThrustToOmega(const Vector<4>& thrusts) const;

  // Getter Functions for member variables
  Matrix<4, 4> getAllocationMatrix() const;

  friend std::ostream& operator<<(std::ostream& os, const Quadrotor& quad);

  // Quadrotor physics
  Scalar m_;
  Matrix<3, 4> t_BM_;
  Matrix<3, 3> J_;
  Matrix<3, 3> J_inv_;

  // Motor
  Scalar motor_omega_min_;
  Scalar motor_omega_max_;
  Scalar motor_tau_inv_;

  // Propellers
  Vector<3> thrust_map_;
  Vector<3> torque_map_;
  Scalar kappa_;
  Scalar thrust_min_;
  Scalar thrust_max_;
  RotorConfig rotors_config_;

  // Quadrotor limits
  Vector<3> omega_max_;

  // Simple cubic aerodynamic model f= c_1 * v +_ c_3 * v^3.
  Vector<3> aero_coeff_1_;
  Vector<3> aero_coeff_3_;

  // Forward-flight induced thrust variation fz -= c_h * (vx^2 + vy^2)
  Scalar aero_coeff_h_;
};

}  // namespace agi
