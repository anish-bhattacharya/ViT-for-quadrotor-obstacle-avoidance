#pragma once

#include "dodgelib/math/math.hpp"
#include "dodgelib/simulator/model_base.hpp"

namespace agi {

class ModelMotor : public ModelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ModelMotor(Quadrotor quad) : ModelBase(quad) {}

  bool run(const Ref<const Vector<QS::SIZE>>,
           Ref<Vector<QS::SIZE>>) const override;
};


}  // namespace agi
