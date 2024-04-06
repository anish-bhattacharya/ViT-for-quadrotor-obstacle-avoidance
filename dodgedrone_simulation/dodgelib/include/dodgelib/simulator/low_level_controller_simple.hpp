#pragma once

#include <iostream>

#include "dodgelib/simulator/low_level_controller_base.hpp"

namespace agi {

class LowLevelControllerSimple : public LowLevelControllerBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LowLevelControllerSimple(Quadrotor quad);
  bool setCommand(const Command& cmd) override;
  void run() override;
  bool updateQuad(const Quadrotor& quad) override;

 private:
  // Quadrotor properties
  Matrix<4, 4> B_allocation_;
  Matrix<4, 4> B_allocation_inv_;

  // P gain for body rate control
  const Matrix<3, 3> Kinv_ang_vel_tau_ =
    Vector<3>(20.0, 20.0, 40.0).asDiagonal();
};


}  // namespace agi
