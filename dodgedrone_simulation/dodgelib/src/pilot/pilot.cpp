#include "dodgelib/pilot/pilot.hpp"

#include "dodgelib/estimator/feedthrough/feedthrough_estimator.hpp"
#include "dodgelib/reference/hover_reference.hpp"
#include "dodgelib/reference/trajectory_reference/polynomial_trajectory.hpp"
#include "dodgelib/reference/trajectory_reference/sampled_trajectory.hpp"
#include "dodgelib/utils/logger.hpp"
#include "dodgelib/utils/throttler.hpp"

namespace agi {

Pilot::Pilot(const PilotParams& params, const TimeFunction time_function)
  : time_(time_function), params_(params) {
  // Load modules in pipeline.
  params_.createPipeline(&pipeline_);

  debug_bridge_ = std::make_shared<DebugBridge>("DebugBridge", time_function);
  if (params_.createBridge(bridge_, time_function)) {
    registerFeedbackCallbacks(bridge_);
  } else {
    logger_.warn(
      "Did not create bridge '%s'.\nUsing debug bridge, register externally!",
      params_.pipeline_cfg_.bridge_cfg.type.c_str());
    bridge_ = debug_bridge_;
  }

  pipeline_.bridge_ = debug_bridge_;
}

Pilot::~Pilot() {
  if (pipeline_thread_.joinable()) {
    shutdown_ = true;
    pipeline_thread_.join();
  }
  shutdown_ = true;
}

void Pilot::launchPipeline() {
  pipeline_thread_ = std::thread(&Pilot::pipelineThread, this);
}

void Pilot::runPipeline() { runPipeline(time_()); }

void Pilot::runPipeline(const Scalar t) { pipeline_.run(time_()); }

bool Pilot::addHover(const Vector<3>& hover_pos, Scalar yaw, Scalar start_time,
                     Scalar duration) {
  logger_.info("Add Hover reference.");
  const QuadState state = pipeline_.getState();
  if (!std::isfinite(yaw)) yaw = state.getYaw();
  if (!std::isfinite(yaw)) yaw = 0.0;
  if (!std::isfinite(start_time))
    start_time = pipeline_.references_.back()->getEndTime();
  if (!std::isfinite(start_time)) start_time = state.t;
  if (!std::isfinite(duration)) duration = INF;

  QuadState hover_state(start_time, hover_pos, yaw);
  pipeline_.appendReference(std::make_shared<HoverReference>(hover_state));

  return true;
}

bool Pilot::setVelocityReference(const Vector<3>& velocity,
                                 const Scalar yaw_rate) {
  static std::shared_ptr<VelocityReference> reference;
  if (isInVelocityReference()) {
    if (!reference->update(velocity, yaw_rate)) {
      logger_.warn("Could not update velocity reference!");
      return false;
    }
  } else {
    if (pipeline_.references_.size() > 1) {
      logger_.warn("More references set, won't switch to velocity reference!");
      return false;
    }
    if (!isInHover()) {
      logger_.warn("Not in hover, won't switch to velocity reference!");
      return false;
    }

    QuadState state;
    if (!pipeline_.estimator_->getRecent(&state)) {
      logger_.warn(
        "Could not get reference state, wont't switch to velocity reference!");
      return false;
    }

    reference = std::make_shared<VelocityReference>(state);
    reference->update(velocity, yaw_rate);
    pipeline_.appendReference(reference);
  }

  return true;
}

bool Pilot::isInHover() const {
  return pipeline_.references_.front()->isHover();
}

bool Pilot::isInVelocityReference() const {
  return pipeline_.references_.front()->isVelocityRefernce();
}

bool Pilot::start() {
  if (!pipeline_.isEstimatorSet()) {
    logger_.error("No estimator set, will not start!");
    return false;
  }

  QuadState curr_state;
  if (!pipeline_.estimator_->getRecent(&curr_state)) {
    logger_.error("No valid state estimate, will not start!");
    return false;
  }

  QuadState state = curr_state.getHoverState();

  if (curr_state.p[2] > params_.takeoff_threshold_) {
    logger_.info(
      "Z-position larger than takeoff threshold, assuming handheld start!");
    pipeline_.appendReference(std::make_shared<HoverReference>(state));
  } else {
    logger_.info("Z-position smaller than takeoff height, taking off!");
    QuadState hover_state = state;
    hover_state.t += params_.takeoff_heigth_ / params_.start_land_speed_;
    hover_state.p[2] += params_.takeoff_heigth_;
    MinSnapTrajectory start_trajectory(curr_state, hover_state);
    if (!start_trajectory.valid()) {
      logger_.warn("Could not find start trajectory!");
      return false;
    }
    pipeline_.appendReference(
      std::make_shared<MinSnapTrajectory>(start_trajectory));
  }

  return true;
}

bool Pilot::land() {
  Pipeline& active_pipeline = getActivePipeline();
  // check if we are close to hover
  if (!active_pipeline.references_.front()->isHover()) {
    logger_.warn(
      "Cannot land (yet) when not in hover! Initiating force hover!");
    forceHover();
    return false;
  }

  // compute landing trajectory
  const QuadState current_hover_state =
    active_pipeline.getState().getHoverState();
  if (!current_hover_state.valid()) {
    logger_.warn("Could not get valid hover state for landing!");
    return false;
  }

  QuadState land_state = current_hover_state;
  land_state.p.z() = 0.0;
  land_state.t += std::abs(land_state.p.z() - current_hover_state.p.z()) /
                  params_.start_land_speed_;
  MinJerkTrajectory land_trajectory(current_hover_state, land_state);
  active_pipeline.insertReference(
    std::make_shared<MinJerkTrajectory>(current_hover_state, land_state));
  return true;
}

bool Pilot::forceHover() {
  Pipeline& active_pipeline = getActivePipeline();
  const QuadState curr_state = active_pipeline.getState();
  if (!curr_state.valid()) {
    logger_.warn("Could not get valid state for force hover!");
    return false;
  }

  active_pipeline.insertReference(std::make_shared<HoverReference>(curr_state));

  return true;
}

bool Pilot::goToPose(const QuadState& end_state) {
  Pipeline& active_pipeline = getActivePipeline();
  const QuadState curr_state = active_pipeline.getState();

  if (!curr_state.valid()) {
    logger_.warn("Could not get valid state for go-to-pose!");
    return false;
  }

  // Set minimum duration of 1.5 seconds to avoid big jumps in the reference
  const Scalar duration = std::max(
    (end_state.p - curr_state.p).norm() / params_.go_to_pose_mean_vel_, 1.5);
  addPolynomialTrajectory(end_state.getHoverState(), duration);
  return true;
}

bool Pilot::off() {
  pipeline_.bridge_->deactivate();
  pipeline_.references_.clear();
  pipeline_.bridge_->reset();

  return true;
}

bool Pilot::appendTrajectory(const QuadState& start_state,
                             const QuadState& end_state) {
  Pipeline& active_pipeline = getActivePipeline();
  if (params_.traj_type_ == "poly_min_snap") {
    active_pipeline.appendReference(
      std::make_shared<MinSnapTrajectory>(start_state, end_state));
  } else if (params_.traj_type_ == "poly_min_jerk") {
    active_pipeline.appendReference(
      std::make_shared<MinJerkTrajectory>(start_state, end_state));
  } else {
    logger_.info(
      "# Trajectory type param did not match any trajectory candidate: %s",
      params_.traj_type_.c_str());
    return false;
  }

  return true;
}

bool Pilot::addPolynomialTrajectory(const QuadState& end_state,
                                    const Scalar duration) {
  logger_ << std::string(80, '=') << std::endl;
  logger_.info("Add polynomial trajectory.");
  Pipeline& active_pipeline = getActivePipeline();
  if (active_pipeline.references_.empty()) {
    logger_.error("Cannot add trajectory when not in flight yet!");
    return false;
  }

  if (!end_state.valid()) {
    logger_.error("End state is not valid!");
    return false;
  }

  // generate fully constrained polynomial trajectory
  QuadState end_state_prev_traj =
    active_pipeline.references_.back()->getEndSetpoint().state.getHoverState();

  const Scalar t_curr = active_pipeline.getState().t;
  if (!std::isfinite(end_state_prev_traj.t) || end_state_prev_traj.t < t_curr)
    end_state_prev_traj.t = t_curr;

  QuadState end_state_poly_traj = end_state;
  end_state_poly_traj.t = end_state_prev_traj.t + duration;

  logger_.info("computing trajectory from t = %.2f to t = %.2f",
               end_state_prev_traj.t, end_state_poly_traj.t);

  if (!appendTrajectory(end_state_prev_traj, end_state_poly_traj)) {
    return false;
  }

  return true;
}


bool Pilot::addPolynomialTrajectory(const QuadState& start,
                                    const QuadState& end,
                                    const std::vector<QuadState>& waypoints,
                                    const Scalar speed,
                                    const Scalar limit_scale) {
  const int n = (int)waypoints.size();
  logger_.info("Got %d waypoints!", n);

  if (!start.p.allFinite()) {
    logger_.warn("Need valid start position!");
    return false;
  }
  if (!end.p.allFinite()) {
    logger_.warn("Need valid end position!");
    return false;
  }

  std::vector<QuadState> states;
  states.reserve(2 + n);
  QuadState start_state;
  QuadState end_state;

  const Scalar t_last = pipeline_.references_.back()->getEndTime();

  start_state.t = std::max(start.t, std::isfinite(t_last) ? t_last : start.t);

  start_state.p = start.p;
  start_state.v = start.v.array().isFinite().select(start.v, 0.0);
  start_state.a = start.a.array().isFinite().select(start.a, 0.0);
  start_state.q(start.q().coeffs().allFinite() ? start.q()
                                               : Quaternion(1, 0, 0, 0));
  start_state.w = start.w.array().isFinite().select(start.w, 0.0);

  end_state.p = end.p;
  end_state.v = end.v.array().isFinite().select(end.v, 0.0);
  end_state.a = end.a.array().isFinite().select(end.a, 0.0);
  end_state.q(end.q().coeffs().allFinite() ? end.q() : Quaternion(1, 0, 0, 0));
  end_state.w = end.w.array().isFinite().select(end.w, 0.0);

  int order = 13;
  Vector<> dists = Vector<>::Zero(n + 1);

  {
    Vector<3> last_pos = start.p;
    for (int i = 0; i < n; ++i) {
      const Vector<> p = waypoints[i].p;
      if (!p.allFinite()) {
        logger_.warn("Need valid waypoint positions!");
        return false;
      }
      dists(i) = dists(std::max(0, i - 1)) + (p - last_pos).norm();
      last_pos = p;
    }
    dists(n) = dists(n - 1) + (end.p - last_pos).norm();
  }

  const Vector<> t = start_state.t + (1.0 / speed) * dists.array();

  logger_.info("Total duration: %1.3fs", t(n) - t(0));

  logger_ << "Adding start state:" << std::endl << start_state;
  states.push_back(start_state);

  for (int i = 0; i < n; ++i) {
    const QuadState& state = waypoints[i];
    QuadState new_state;
    new_state.t = t(i);
    new_state.p = state.p;
    new_state.v = state.v;
    new_state.w = state.w;
    new_state.q(state.q());

    order += (int)state.p.allFinite() + (int)state.v.allFinite();
    logger_ << "Adding waypoint " << i << ":" << std::endl << new_state;
    states.push_back(new_state);
  }

  end_state.t = t(n);
  logger_ << "Adding end state:" << std::endl << end_state;
  states.push_back(end_state);

  logger_.info("Trying to compute with order %d", order);
  const Vector<4> weights(0.0, 0.0, 0.0, 1.0);
  PolynomialTrajectory<> traj(states, weights, order);
  if (!traj.valid()) {
    logger_.warn("Could not compute trajectory through waypoints!");
    return false;
  }

  if (limit_scale > 0.0) {
    logger_.info("Rescaling to quad limit scaled by %1.3f", limit_scale);
    logger_.info("Start at %1.3fs", traj.getDuration());
    traj.scaleToLimits(limit_scale * params_.quad_.collective_thrust_max() /
                       params_.quad_.m_);
    logger_.info("Rescaling to %1.3fs", traj.getDuration());
  }

  if (!pipeline_.appendReference(
        std::make_shared<PolynomialTrajectory<>>(traj))) {
    logger_.warn("Could not append trajectory through waypoints!");
    return false;
  }

  return true;
}


bool Pilot::addSampledTrajectory(const SetpointVector& setpoints) {
  logger_.info("Add sampled trajectory.");
  if (setpoints.empty()) {
    logger_.error("Received empty trajectory!");
    return false;
  }

  pipeline_.appendReference(std::make_shared<SampledTrajectory>(setpoints));
  return true;
}

bool Pilot::setFeedthroughCommand(const Command& command) {
  return pipeline_.setFeedthroughCommand(command);
}

bool Pilot::getReference(const int idx, ReferenceBase* const reference) const {
  if (reference == nullptr) return false;
  return false;
}

bool Pilot::getActiveReference(const QuadState& curr_state,
                               SetpointVector* const setpoints) const {
  if (!pipeline_.isReferenceSet()) return false;
  return pipeline_.sampler_->getAt(curr_state, pipeline_.references_,
                                   setpoints);
}

int Pilot::getAllReferences(ReferenceVector* const references) const {
  references->clear();
  for (auto& reference : pipeline_.references_) {
    references->push_back(reference);
  }
  return references->size();
}

bool Pilot::registerExternalEstimator(
  const std::shared_ptr<EstimatorBase>& estimator) {
  if (!estimator) return false;

  if (pipeline_.bridge_->active()) return false;

  pipeline_.estimator_ = estimator;

  return true;
}

bool Pilot::registerExternalBridge(const std::shared_ptr<BridgeBase>& bridge) {
  if (!bridge) return false;

  const bool was_used = pipeline_.bridge_ == bridge_;
  const bool was_active = was_used && pipeline_.bridge_->active();

  logger_.info("Register external bridge: %s\n which was %sactive and %sused.",
               bridge->name().c_str(), was_active ? "" : "not ",
               was_used ? "" : "not ");

  if (was_active) pipeline_.bridge_->deactivate();

  bridge_ = bridge;

  if (was_used) pipeline_.bridge_ = bridge_;
  if (was_active) pipeline_.bridge_->activate();

  return true;
}

bool Pilot::registerExternalDebugBridge(
  const std::shared_ptr<BridgeBase>& bridge) {
  if (!bridge) return false;

  const bool was_used = pipeline_.bridge_ == debug_bridge_;

  debug_bridge_ = bridge;

  if (was_used) pipeline_.bridge_ = debug_bridge_;

  return true;
}

bool Pilot::registerFeedbackCallbacks(
  const std::shared_ptr<BridgeBase>& bridge) const {
  if (!bridge) return false;
  bridge->registerFeedbackCallback([&](const Feedback& feedback) {
    pipeline_.estimator_->addImu(feedback.imu);
    if (feedback.rotor_speed_rads.allFinite())
      pipeline_.estimator_->addMotorSpeeds(feedback.rotor_speed_rads);
    pipeline_.outer_controller_->addImuSample(feedback.imu);
    if (pipeline_.isInnerControllerSet())
      pipeline_.inner_controller_->addImuSample(feedback.imu);
  });

  return true;
}

bool Pilot::registerFeedbackCallback(const FeedbackCallbackFunction& function) {
  if (!bridge_) return false;
  bridge_->registerFeedbackCallback(function);

  return true;
}

void Pilot::registerPipelineCallback(
  const Pipeline::PipelineCallbackFunction& function) {
  pipeline_.registerCallback(function);
}

void Pilot::odometryCallback(const Pose& pose) {
  pipeline_.estimator_->addPose(pose);
}

void Pilot::odometryCallback(const QuadState& state) {
  if (params_.velocity_in_bodyframe_) {
    QuadState state_world_frame = state;
    state_world_frame.v = state.q() * state.v;
    pipeline_.estimator_->addState(state_world_frame);
  } else {
    pipeline_.estimator_->addState(state);
  }
}

void Pilot::voltageCallback(const Scalar voltage) {
  bridge_->setVoltage(voltage);
}

void Pilot::imuCallback(const ImuSample& imu) {
  if (pipeline_.estimator_) pipeline_.estimator_->addImu(imu);
  if (pipeline_.outer_controller_)
    pipeline_.outer_controller_->addImuSample(imu);
  if (pipeline_.inner_controller_)
    pipeline_.inner_controller_->addImuSample(imu);
}

void Pilot::motorSpeedCallback(const Vector<4>& mot) {
  if (pipeline_.estimator_) pipeline_.estimator_->addMotorSpeeds(mot);
}

void Pilot::pipelineThread() {
  while (!shutdown_) {
    pipeline_timer_.tic();
    pipeline_.run(time_());

    const Scalar dt_last = pipeline_timer_.toc();

    const Scalar remainder = dt_last - params_.dt_min_;
    if (remainder < 0.0) {
      std::this_thread::sleep_for(
        std::chrono::nanoseconds((int)(-1e9 * remainder)));
    } else {
      logger_.info("Pipeline too slow by %1.3fms with %1.3fms!",
                   1e3 * remainder, 1e3 * dt_last);
    }
  }
}

Command Pilot::getCommand() const { return pipeline_.getCommand(); }

const SetpointVector Pilot::getOuterSetpoints() const {
  return pipeline_.getOuterSetpoints();
}

const SetpointVector Pilot::getInnerSetpoints() const {
  return pipeline_.getInnerSetpoints();
}

QuadState Pilot::getRecentState() const {
  QuadState state;
  if (!pipeline_.estimator_->getRecent(&state))
    logger_.warn("Could not get recent state!");
  return state;
}

Scalar Pilot::getVoltage() const { return bridge_->getVoltage(); }

void Pilot::enable(bool enable) {
  if (enable) {
    pipeline_.bridge_->deactivate();
    if (!bridge_->active()) bridge_->activate();
    pipeline_.bridge_ = bridge_;
  } else {
    pipeline_.bridge_->deactivate();
    pipeline_.bridge_ = debug_bridge_;
    pipeline_.bridge_->activate();
  }
}

bool Pilot::enabled() {
  return pipeline_.bridge_ != debug_bridge_ && pipeline_.bridge_->active();
}

std::string Pilot::getActiveBridgeType() { return pipeline_.bridge_->name(); }

bool Pilot::getFeedback(Feedback* const feedback) const {
  return bridge_->getFeedback(feedback);
}

bool Pilot::getQuadrotor(Quadrotor* const quad) const {
  if (!params_.quad_.valid()) return false;
  *quad = params_.quad_;
  return true;
}

Pipeline& Pilot::getActivePipeline() { return pipeline_; }

}  // namespace agi
