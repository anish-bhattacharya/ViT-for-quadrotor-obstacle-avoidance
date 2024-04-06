#pragma once

#include "dodgelib/math/gravity.hpp"
#include "dodgelib/simulator/model_base.hpp"

namespace agi {

class ModelThrustTorqueSimple : public ModelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ModelThrustTorqueSimple(Quadrotor quad);

  bool updateQuad(const Quadrotor&) override;
  bool run(const Ref<const Vector<QS::SIZE>>,
           Ref<Vector<QS::SIZE>>) const override;

 private:
  Matrix<4, 4> B_allocation_;
};


}  // namespace agi
