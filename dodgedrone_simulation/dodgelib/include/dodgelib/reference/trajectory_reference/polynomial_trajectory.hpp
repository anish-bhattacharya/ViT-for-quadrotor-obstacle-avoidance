
#pragma once

#include "dodgelib/reference/reference_base.hpp"
#include "dodgelib/reference/trajectory_reference/polynomial.hpp"
#include "dodgelib/types/quadrotor.hpp"

namespace agi {

template<class PolyType = Polynomial<>>
class PolynomialTrajectory : public ReferenceBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PolynomialTrajectory(const QuadState& start_state, const QuadState& end_state,
                       const Vector<>& weights = Vector<4>(0, 0, 0, 1),
                       const int order = 11, const int continuity = -1,
                       const std::string& name = "Polynomial Trajectory");
  PolynomialTrajectory(const std::vector<QuadState>& states,
                       const Vector<>& weights = Vector<4>(0, 0, 0, 1),
                       const int order = 11,
                       const std::string& name = "Polynomial Trajectory");
  virtual ~PolynomialTrajectory() = default;

  using ReferenceBase::getSetpoint;
  Setpoint getSetpoint(const QuadState& state, const Scalar time) override;
  QuadState getState(const Scalar time) const;
  virtual Setpoint getStartSetpoint() override final;
  virtual Setpoint getEndSetpoint() override final;

  bool addStateConstraint(const QuadState& state);
  bool solved() const;
  bool valid() const override;

  // utilities
  Vector<> evalTranslation(const Scalar time, const int order = 0) const;
  Scalar findTimeMaxAcc(const Scalar precision = 1e-3) const;
  Scalar findTimeMaxOmega(const Scalar precision = 1e-3) const;
  Scalar findTimeMaxAcc(const Scalar t_start, const Scalar t_end,
                        const Scalar precision = 1e-3) const;
  Scalar findTimeMaxOmega(const Scalar t_start, const Scalar t_end,
                          const Scalar precision = 1e-3) const;
  void scale(const Scalar start_time = NAN, const Scalar duration = NAN);
  Scalar scaleToLimits(const Quadrotor& quad, const int iterations = 10,
                       const Scalar tolerance = 1e-4);
  Scalar scaleToLimits(const Scalar acc_limit, const int iterations = 10,
                       const Scalar tolerance = 1e-4);

  void setForwardHeading(const bool forward);

 protected:
  template<typename EvalFunc, typename CompFunc>
  Scalar findTime(const Scalar dt, const Scalar dt_min, const Scalar t_start,
                  const Scalar t_end, EvalFunc eval, CompFunc comp) const;

  QuadState end_state_;
  PolyType x_;
  PolyType y_;
  PolyType z_;
  PolyType yaw_;

  std::vector<QuadState> states_;

  mutable Quaternion q_pitch_roll_last_{1, 0, 0, 0};
  mutable Scalar yaw_last_{0.0};
  bool forward_heading_{false};
};

// Child classes
class MinSnapTrajectory : public PolynomialTrajectory<Polynomial<>> {
 public:
  MinSnapTrajectory(const QuadState& start_state, const QuadState& end_state,
                    const int order = 11, const int continuity = 3,
                    const std::string& name = "Minimum Snap Trajectory")
    : PolynomialTrajectory(start_state, end_state, Vector<4>(0, 0, 0, 1), order,
                           continuity, name) {}
};

class MinJerkTrajectory : public PolynomialTrajectory<Polynomial<>> {
 public:
  MinJerkTrajectory(const QuadState& start_state, const QuadState& end_state,
                    const int order = 11, const int continuity = 2,
                    const std::string& name = "Minimum Jerk Trajectory")
    : PolynomialTrajectory(start_state, end_state, Vector<3>(0, 0, 1), order,
                           continuity, name) {}
};

// Closed-Form Minimum-Jerk Specialization
template<>
PolynomialTrajectory<ClosedFormMinJerkAxis>::PolynomialTrajectory(
  const QuadState& start_state, const QuadState& end_state,
  const Vector<>& weights, const int order, const int continuity,
  const std::string& name);

template<>
PolynomialTrajectory<ClosedFormMinJerkAxis>::PolynomialTrajectory(
  const std::vector<QuadState>& states, const Vector<>& weights,
  const int order, const std::string& name) = delete;

class ClosedFormMinJerkTrajectory
  : public PolynomialTrajectory<ClosedFormMinJerkAxis> {
 public:
  ClosedFormMinJerkTrajectory(const QuadState& start_state,
                              const QuadState& end_state)
    : PolynomialTrajectory(start_state, end_state, Vector<3>(0, 0, 1), 5, 3,
                           "Closed-Form Mininum Jerk Trajectory") {}
};

}  // namespace agi
