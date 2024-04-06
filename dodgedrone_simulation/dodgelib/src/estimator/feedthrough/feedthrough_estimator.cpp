#include "dodgelib/estimator/feedthrough/feedthrough_estimator.hpp"

#include "dodgelib/math/math.hpp"

namespace agi {

FeedthroughEstimator::FeedthroughEstimator(
  const std::shared_ptr<FeedthroughParameters>& params)
  : EstimatorBase("Feedthrough Estimator") {
  if (!params) {
    params_ = std::make_shared<FeedthroughParameters>();
  } else {
    params_ = params;
  }
}

bool FeedthroughEstimator::initialize(const QuadState& state) {
  const Vector<4> keep_motor_speeds = state_.mot;
  state_ = state;
  if (keep_motor_speeds.allFinite()) state_.mot = keep_motor_speeds;
  return state_.valid();
}

bool FeedthroughEstimator::addState(const QuadState& state) {
  state_ = params_->transform_enabled_ ? transform(state) : state;
  return state_.valid();
}

bool FeedthroughEstimator::addMotorSpeeds(const Vector<4>& speeds) {
  state_.mot = !speeds.allFinite() ? Vector<4>::Zero() : speeds;
  return true;
}

bool FeedthroughEstimator::addImu(const ImuSample& imu) {
  state_.w = imu.omega;
  return state_.valid();
}

bool FeedthroughEstimator::getAt(const Scalar t, QuadState* const state) {
  if (state == nullptr) return false;
  if (state_.valid()) {
    *state = state_;
    return true;
  } else {
    return false;
  }
}

QuadState FeedthroughEstimator::transform(const QuadState& state_in) {
  const Eigen::AngleAxisd rollAngle(params_->roll_, Vector<3>::UnitX());
  const Eigen::AngleAxisd pitchAngle(params_->pitch_, Vector<3>::UnitY());
  const Eigen::AngleAxisd yawAngle(params_->yaw_, Vector<3>::UnitZ());
  const Quaternion q_inv = (rollAngle * pitchAngle * yawAngle).inverse();

  QuadState state_out(state_in);
  state_out.p = state_in.p + params_->pos_offset_;
  state_out.v = q_inv * state_in.v;
  state_out.qx = Q_right(q_inv) * state_in.qx;
  state_out.w = q_inv * state_in.w;

  return state_out;
}

bool FeedthroughEstimator::healthy() const { return state_.valid(); }

}  // namespace agi
