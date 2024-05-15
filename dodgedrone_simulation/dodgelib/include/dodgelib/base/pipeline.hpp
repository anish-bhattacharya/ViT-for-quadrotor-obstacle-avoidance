#pragma once

#include <memory>

#include "dodgelib/base/module.hpp"
#include "dodgelib/bridge/bridge_base.hpp"
#include "dodgelib/controller/controller_base.hpp"
#include "dodgelib/estimator/estimator_base.hpp"
#include "dodgelib/reference/reference_base.hpp"
#include "dodgelib/sampler/sampler_base.hpp"
#include "dodgelib/types/command.hpp"
#include "dodgelib/types/quad_state.hpp"
#include "dodgelib/types/setpoint.hpp"
#include "dodgelib/utils/logger.hpp"

namespace agi {

class Pipeline {
 public:
  Pipeline(const std::string& name = "Pipeline",
           const Scalar feedthrough_timeout = 0.05);

  using PipelineCallbackFunction = std::function<void(
    const QuadState&, const Feedback&, const ReferenceVector&,
    const SetpointVector&, const SetpointVector&, const SetpointVector&,
    const Command&)>;

  ReferenceVector references_;
  std::shared_ptr<EstimatorBase> estimator_;
  std::shared_ptr<SamplerBase> sampler_;
  std::shared_ptr<ControllerBase> outer_controller_;
  std::shared_ptr<ControllerBase> inner_controller_;
  std::shared_ptr<BridgeBase> bridge_;

  Pipeline& operator=(const Pipeline& other);

  bool isReferenceSet() const;
  bool isEstimatorSet() const;
  bool isSamplerSet() const;
  bool isOuterControllerSet() const;
  bool isInnerControllerSet() const;
  bool isBridgeSet() const;

  bool isSet() const;

  bool appendReference(std::shared_ptr<ReferenceBase>&& reference);
  bool insertReference(std::shared_ptr<ReferenceBase>&& reference);
  bool run(const Scalar t);
  Command getCommand() const;

  bool setFeedthroughCommand(const Command& command);
  void clearFeedthoughCommand();
  inline bool feedthroughActive() const { return feedthrough_active_; }

  inline const QuadState& getState() const { return state_; }
  inline const SetpointVector getSetpoints() const { return setpoints_; }
  inline const SetpointVector& getOuterSetpoints() const {
    return setpoints_outer_;
  }
  inline const SetpointVector& getInnerSetpoints() const {
    return setpoints_inner_;
  }

  void printReferenceInfo(const bool detailed = false) const;
  void registerCallback(const PipelineCallbackFunction& function);
  void setOuterloopDivisor(const int divisor);
  void setStopAfterFeedthrough(const bool stop_after_feedthrough);
  void setFeedthroughTimeout(const Scalar& feedthrough_timeout);

  [[nodiscard]] Scalar dt() const { return dt_; }

 private:
  QuadState state_;
  Feedback feedback_;
  SetpointVector setpoints_;
  SetpointVector setpoints_outer_;
  SetpointVector setpoints_inner_;
  Command command_;
  Command apply_command_;
  Command feedthrough_command_;
  Logger logger_;

  bool initialized_{false};
  bool feedthrough_active_{false};
  Scalar dt_{0.01};
  int outerloop_divisor_{1};
  int outerloop_counter_{-1};
  bool stop_after_feedthrough_;
  Scalar feedthrough_timeout_{0.05};

  std::vector<PipelineCallbackFunction> callbacks_;
};

}  // namespace agi
