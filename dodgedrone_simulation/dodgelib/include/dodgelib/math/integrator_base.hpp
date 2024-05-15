#pragma once

#include <functional>

#include "dodgelib/math/types.hpp"
#include "dodgelib/types/quad_state.hpp"

namespace agi {

class IntegratorBase {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 public:
  IntegratorBase(DynamicsFunction function, const Scalar dt_max = 1e-3);

  virtual ~IntegratorBase() = default;

  bool integrate(const QuadState& initial_state,
                 QuadState* const final_state) const;

  bool integrate(const Ref<const Vector<>> initial_state, const Scalar dt,
                 Ref<Vector<>> final_state) const;

  virtual bool step(const Ref<const Vector<>> initial, const Scalar dt,
                    Ref<Vector<>> final_state) const = 0;

  Scalar dtMax() const;

 protected:
  DynamicsFunction dynamics_;
  Scalar dt_max_;
};

}  // namespace agi
