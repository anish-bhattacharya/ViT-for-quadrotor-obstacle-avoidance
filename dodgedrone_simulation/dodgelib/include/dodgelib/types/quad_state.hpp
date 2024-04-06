#pragma once

#include <memory>

#include "dodgelib/math/types.hpp"

namespace agi {

struct QuadState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum IDX : int {
    POS = 0,
    POSX = 0,
    POSY = 1,
    POSZ = 2,
    NPOS = 3,
    ATT = 3,
    ATTW = 3,
    ATTX = 4,
    ATTY = 5,
    ATTZ = 6,
    NATT = 4,
    VEL = 7,
    VELX = 7,
    VELY = 8,
    VELZ = 9,
    NVEL = 3,
    OME = 10,
    OMEX = 10,
    OMEY = 11,
    OMEZ = 12,
    NOME = 3,
    ACC = 13,
    ACCX = 13,
    ACCY = 14,
    ACCZ = 15,
    NACC = 3,
    TAU = 16,
    TAUX = 16,
    TAUY = 17,
    TAUZ = 18,
    NTAU = 3,
    JERK = 19,
    JERKX = 19,
    JERKY = 20,
    JERKZ = 21,
    NJERK = 3,
    SNAP = 22,
    SNAPX = 22,
    SNAPY = 23,
    SNAPZ = 24,
    NSNAP = 3,
    BOME = 25,
    BOMEX = 25,
    BOMEY = 26,
    BOMEZ = 27,
    NBOME = 3,
    BACC = 28,
    BACCX = 28,
    BACCY = 29,
    BACCZ = 30,
    NBACC = 3,
    MOT = 31,
    MOT1 = 31,
    MOT2 = 32,
    MOT3 = 33,
    MOT4 = 34,
    NMOT = 4,
    MOTDES = 35,
    MOTDES1 = 35,
    MOTDES2 = 36,
    MOTDES3 = 37,
    MOTDES4 = 38,
    NMOTDES = 4,
    SIZE = 39,
    DYN = 19
  };

  QuadState();
  QuadState(const Scalar t, const Vector<IDX::SIZE>& x);
  QuadState(const Scalar t, const Vector<3>& position, const Scalar yaw);
  QuadState(const QuadState& state);
  ~QuadState();

  inline static int size() { return SIZE; }

  Quaternion q() const;
  void q(const Quaternion quaternion);
  void q(const Scalar angle, const Vector<3>& axis = Vector<3>::UnitZ());
  Matrix<3, 3> R() const;

  void setZero(const bool& reset_time = true);
  void linearize();
  inline bool valid() const { return x.allFinite() && std::isfinite(t); }

  Vector<IDX::SIZE> x = Vector<IDX::SIZE>::Constant(NAN);
  Scalar t{NAN};

  Ref<Vector<3>> p{x.segment<IDX::NPOS>(IDX::POS)};
  Ref<Vector<4>> qx{x.segment<IDX::NATT>(IDX::ATT)};
  Ref<Vector<3>> v{x.segment<IDX::NVEL>(IDX::VEL)};
  Ref<Vector<3>> w{x.segment<IDX::NOME>(IDX::OME)};
  Ref<Vector<3>> a{x.segment<IDX::NACC>(IDX::ACC)};
  Ref<Vector<3>> tau{x.segment<IDX::NTAU>(IDX::TAU)};
  Ref<Vector<3>> j{x.segment<IDX::NJERK>(IDX::JERK)};
  Ref<Vector<3>> s{x.segment<IDX::NSNAP>(IDX::SNAP)};
  Ref<Vector<3>> bw{x.segment<IDX::NBOME>(IDX::BOME)};
  Ref<Vector<3>> ba{x.segment<IDX::NBACC>(IDX::BACC)};
  Ref<Vector<4>> mot{x.segment<IDX::NMOT>(IDX::MOT)};
  Ref<Vector<4>> motdes{x.segment<IDX::NMOTDES>(IDX::MOTDES)};

  Scalar getYaw(const Scalar yaw = NAN) const;
  void applyYaw(const Scalar angle);
  QuadState getHoverState() const;

  bool operator==(const QuadState& rhs) const;
  bool isApprox(const QuadState& rhs, const Scalar tol = 1e-6) const;

  inline bool operator<(const Scalar time) const { return t < time; }
  inline bool operator<=(const Scalar time) const { return t <= time; }
  inline bool operator>(const Scalar time) const { return t > time; }
  inline bool operator>=(const Scalar time) const { return t >= time; }

  friend std::ostream& operator<<(std::ostream& os, const QuadState& state);
};

using QS = QuadState;

}  // namespace agi
