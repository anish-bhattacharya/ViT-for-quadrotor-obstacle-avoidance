#pragma once

#include "dodgelib/math/types.hpp"

namespace agi {

struct ImuSample {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ImuSample() = default;
  ImuSample(const ImuSample&) = default;

  ImuSample(const Scalar t, const Vector<3>& acc, const Vector<3>& omega)
    : t(t), acc(acc), omega(omega) {}

  ~ImuSample() = default;

  inline bool valid() const {
    return omega.allFinite() && acc.allFinite() && std::isfinite(t);
  }

  Scalar t{NAN};
  Vector<3> acc{NAN, NAN, NAN};
  Vector<3> omega{NAN, NAN, NAN};

  inline bool operator<(const Scalar time) const { return t < time; }
  inline bool operator<=(const Scalar time) const { return t <= time; }
  inline bool operator>(const Scalar time) const { return t > time; }
  inline bool operator>=(const Scalar time) const { return t >= time; }
};

}  // namespace agi