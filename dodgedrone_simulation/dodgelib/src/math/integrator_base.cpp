#include "dodgelib/math/integrator_base.hpp"

namespace agi {

IntegratorBase::IntegratorBase(DynamicsFunction function, const Scalar dt_max)
  : dynamics_(function), dt_max_(dt_max) {}

bool IntegratorBase::integrate(const QuadState& initial_state,
                               QuadState* const final_state) const {
  if (final_state == nullptr) return false;
  if (std::isnan(initial_state.t) || std::isnan(final_state->t)) return false;
  if (initial_state.t >= final_state->t) return false;
  return integrate(initial_state.x, final_state->t - initial_state.t,
                   final_state->x);
}

bool IntegratorBase::integrate(const Ref<const Vector<>> initial_state,
                               const Scalar dt,
                               Ref<Vector<>> final_state) const {
  Scalar dt_remaining = dt;
  Vector<> state = initial_state;

  do {
    const Scalar dt_this = std::min(dt_remaining, dt_max_);
    if (!step(state, dt_this, final_state)) return false;
    state = final_state;
    dt_remaining -= dt_this;
  } while (dt_remaining > 0.0);

  return true;
}

Scalar IntegratorBase::dtMax() const { return dt_max_; }

}  // namespace agi
