#pragma once

#include "dodgelib/base/pipeline.hpp"
#include "dodgelib/bridge/debug_bridge.hpp"
#include "dodgelib/pilot/pilot_params.hpp"
#include "dodgelib/reference/velocity_reference.hpp"
#include "dodgelib/types/quadrotor.hpp"
#include "dodgelib/utils/logger.hpp"
#include "dodgelib/utils/timer.hpp"

namespace agi {

class Pilot {
 public:
  Pilot(const PilotParams& params, const TimeFunction time_function);
  ~Pilot();

  void launchPipeline();
  void runPipeline();
  void runPipeline(const Scalar t);


  bool getReference(const int idx, ReferenceBase* const reference) const;
  bool getActiveReference(const QuadState& curr_state,
                          SetpointVector* const setpoints) const;
  int getAllReferences(ReferenceVector* const references) const;

  bool registerExternalEstimator(
    const std::shared_ptr<EstimatorBase>& estimator);
  bool registerExternalBridge(const std::shared_ptr<BridgeBase>& bridge);
  bool registerFeedbackCallbacks(
    const std::shared_ptr<BridgeBase>& bridge) const;
  bool registerFeedbackCallback(const FeedbackCallbackFunction& function);
  bool registerExternalDebugBridge(const std::shared_ptr<BridgeBase>& bridge);
  void registerPipelineCallback(
    const Pipeline::PipelineCallbackFunction& function);

  void odometryCallback(const Pose& pose);
  void odometryCallback(const QuadState& state);
  void voltageCallback(const Scalar voltage);
  void imuCallback(const ImuSample& imu);
  void motorSpeedCallback(const Vector<4>& mot);

  Command getCommand() const;
  const SetpointVector getOuterSetpoints() const;
  const SetpointVector getInnerSetpoints() const;
  QuadState getRecentState() const;
  Scalar getVoltage() const;

  void enable(bool enable);
  bool enabled();

  bool start();
  bool land();
  bool off();

  bool forceHover();
  bool goToPose(const QuadState& end_state);
  bool setVelocityReference(const Vector<3>& velocity, const Scalar yaw_rate);

  bool addHover(const Vector<3>& hover_pos, Scalar yaw = NAN,
                Scalar start_time = NAN, Scalar duration = NAN);

  template<typename ReferenceType>
  bool addReference(const ReferenceType& reference) {
    return pipeline_.appendReference(reference);
  }
  bool appendTrajectory(const QuadState& start_state,
                        const QuadState& end_state);
  bool addPolynomialTrajectory(const QuadState& end_state,
                               const Scalar duration);
  bool addPolynomialTrajectory(const QuadState& start, const QuadState& end,
                               const std::vector<QuadState>& end_state,
                               const Scalar speed,
                               const Scalar limit_scale = 0.0);
  bool addSampledTrajectory(const SetpointVector& setpoints);

  bool setFeedthroughCommand(const Command& command);

  bool isInHover() const;
  bool isInVelocityReference() const;
  std::string getActiveBridgeType();

  bool getFeedback(Feedback* const feedback = nullptr) const;

  bool getQuadrotor(Quadrotor* const quad) const;
  inline const PilotParams& getParams() const { return params_; }
  inline Scalar getTime() const { return time_(); }

 private:
  void pipelineThread();
  Pipeline& getActivePipeline();
  const TimeFunction time_;
  PilotParams params_;
  Pipeline pipeline_;
  std::shared_ptr<BridgeBase> bridge_;
  std::shared_ptr<BridgeBase> debug_bridge_;

  bool shutdown_{false};
  std::thread pipeline_thread_;
  Timer pipeline_timer_;

  Logger logger_{"Pilot"};
};


}  // namespace agi
