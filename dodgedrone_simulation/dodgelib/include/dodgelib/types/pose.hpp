#pragma once

#include "dodgelib/math/types.hpp"

namespace agi {

struct Pose {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Pose() = default;
  Pose(const Pose&) = default;
  ~Pose() = default;

  static constexpr int SIZE = 7;

  bool valid() const {
    return std::isfinite(t) && position.allFinite() &&
           attitude.coeffs().allFinite();
  }

  Scalar t{NAN};
  Vector<3> position{NAN, NAN, NAN};
  Quaternion attitude{NAN, NAN, NAN, NAN};
};


}  // namespace agi
