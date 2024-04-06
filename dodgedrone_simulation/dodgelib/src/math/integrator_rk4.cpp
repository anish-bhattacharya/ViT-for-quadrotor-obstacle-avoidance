#include "dodgelib/math/integrator_rk4.hpp"

namespace agi {

bool IntegratorRK4::step(const Ref<const Vector<>> initial_state,
                         const Scalar dt, Ref<Vector<>> final_state) const {
  static const Vector<4> rk4_sum_vec{1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0,
                                     1.0 / 6.0};
  Matrix<> k = Matrix<>::Zero(initial_state.rows(), 4);

  final_state = initial_state;

  // k_1
  if (!this->dynamics_(final_state, k.col(0))) return false;

  // k_2
  final_state = initial_state + 0.5 * dt * k.col(0);
  if (!this->dynamics_(final_state, k.col(1))) return false;

  // k_3
  final_state = initial_state + 0.5 * dt * k.col(1);
  if (!this->dynamics_(final_state, k.col(2))) return false;

  // k_4
  final_state = initial_state + dt * k.col(2);
  if (!this->dynamics_(final_state, k.col(3))) return false;


  final_state = initial_state + dt * k * rk4_sum_vec;

  return true;
}

}  // namespace agi