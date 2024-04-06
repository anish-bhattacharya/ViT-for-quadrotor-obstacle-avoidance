#pragma once

#include <memory>

#include "dodgelib/estimator/estimator_base.hpp"
#include "dodgelib/estimator/feedthrough/feedthrough_params.hpp"
#include "dodgelib/types/pose.hpp"
#include "dodgelib/types/quad_state.hpp"

namespace agi {

class FeedthroughEstimator : public EstimatorBase {
 public:
  FeedthroughEstimator(const std::shared_ptr<FeedthroughParameters>& params);

  virtual bool initialize(const QuadState& state) override;

  virtual bool addPose(const Pose& pose) override { return false; }
  virtual bool addState(const QuadState& pose) override;
  virtual bool addMotorSpeeds(const Vector<4>& speeds) override;
  virtual bool addImu(const ImuSample& imu) override;

  virtual bool getAt(const Scalar t, QuadState* const state) override;

  virtual bool healthy() const override;

 private:
  QuadState transform(const QuadState& state_in);
  std::shared_ptr<FeedthroughParameters> params_;
  QuadState state_;
};

}  // namespace agi
