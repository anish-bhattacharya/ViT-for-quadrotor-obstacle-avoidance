#include "dodgelib/estimator/estimator_base.hpp"

namespace agi {

EstimatorBase::EstimatorBase(const std::string& name) : Module(name) {}

EstimatorBase::~EstimatorBase() {}

bool EstimatorBase::getState(QuadState* const state) {
  if (state == nullptr) return false;
  return getAt(state->t, state);
}

bool EstimatorBase::getRecent(QuadState* const state) {
  if (state == nullptr) return false;
  return getAt(-1.0, state);
}

QuadState EstimatorBase::getRecent() {
  QuadState state;
  getAt(-1.0, &state);
  return state;
}

}  // namespace agi
