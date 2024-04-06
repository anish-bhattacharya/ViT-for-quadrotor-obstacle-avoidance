#pragma once

#include <memory>

#include "dodgelib/base/module.hpp"
#include "dodgelib/types/imu_sample.hpp"
#include "dodgelib/types/pose.hpp"
#include "dodgelib/types/quad_state.hpp"

namespace agi {

class EstimatorBase : public Module<EstimatorBase> {
 public:
  EstimatorBase(const std::string& name = "Estimator");
  virtual ~EstimatorBase();

  /// Initialize the filter.
  virtual bool initialize(const QuadState& state) = 0;

  /// Add measurement
  virtual bool addPose(const Pose& pose) = 0;
  virtual bool addState(const QuadState& pose) = 0;
  virtual bool addImu(const ImuSample& pose) = 0;
  virtual bool addMotorSpeeds(const Vector<4>& speeds) = 0;

  /// Get state at specific time.
  virtual bool getAt(const Scalar t, QuadState* const state) = 0;

  /// Check if estimator is healthy.
  virtual bool healthy() const = 0;

  /// Get state at time state.t.
  bool getState(QuadState* const state);

  /// Get latest state.
  bool getRecent(QuadState* const state);
  QuadState getRecent();
};

}  // namespace agi