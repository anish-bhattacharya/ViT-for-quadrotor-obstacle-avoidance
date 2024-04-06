#include "dodgelib/base/pipeline.hpp"

#include "dodgelib/reference/hover_reference.hpp"

namespace agi {

Pipeline::Pipeline(const std::string& name, const Scalar feedthrough_timeout)
  : logger_(name), feedthrough_timeout_(feedthrough_timeout) {}

Pipeline& Pipeline::operator=(const Pipeline& other) {
  references_ = other.references_;
  estimator_ = other.estimator_;
  sampler_ = other.sampler_;
  outer_controller_ = other.outer_controller_;
  inner_controller_ = other.inner_controller_;
  bridge_ = other.bridge_;
  return *this;
}

bool Pipeline::isReferenceSet() const { return !references_.empty(); }

bool Pipeline::isEstimatorSet() const { return estimator_ != nullptr; }

bool Pipeline::isSamplerSet() const { return sampler_ != nullptr; }

bool Pipeline::isOuterControllerSet() const {
  return outer_controller_ != nullptr;
}

bool Pipeline::isInnerControllerSet() const {
  return inner_controller_ != nullptr;
}

bool Pipeline::isBridgeSet() const { return bridge_ != nullptr; }

bool Pipeline::isSet() const {
  return !references_.empty() && estimator_ && sampler_ && outer_controller_ &&
         bridge_;
}

bool Pipeline::setFeedthroughCommand(const Command& command) {
  if (!command.valid()) return false;

  feedthrough_command_ = command;
  return true;
}

void Pipeline::clearFeedthoughCommand() { feedthrough_command_ = Command(); }

Command Pipeline::getCommand() const {
  if (!bridge_->active()) {
    return Command();
  }

  return apply_command_;
}

bool Pipeline::run(const Scalar t) {
  // Get State
  if (estimator_) {
    const bool estimator_successful = estimator_->getAt(t, &state_);
    initialized_ |= estimator_successful;
    if (initialized_ && !estimator_successful) {
      logger_.error("Estimator failed!");
      return false;
    }
  }

  if (references_.empty()) {
    command_ = Command(t);
  } else {
    // Get Setpoints
    if (!sampler_ || !sampler_->getAt(state_, references_, &setpoints_)) {
      logger_.error("Sampler failed!");
      return false;
    }

    if (outerloop_divisor_ < 2 ||
        !(++outerloop_counter_ % outerloop_divisor_)) {
      outerloop_counter_ = 0;
      if (!isOuterControllerSet() || !outer_controller_->getCommand(
                                       state_, setpoints_, &setpoints_outer_)) {
        logger_.error("High Level Controller failed!");
        return false;
      } else {
        outer_controller_->logTiming();
      }
    }

    if (isInnerControllerSet()) {
      if (!inner_controller_->getCommand(state_, setpoints_outer_,
                                         &setpoints_inner_)) {
        logger_.error("Low Level Controller failed!");
        return false;
      }
    } else {
      setpoints_inner_.clear();
    }

    if (!setpoints_inner_.empty()) {
      command_ = setpoints_inner_.front().input;
    } else if (!setpoints_outer_.empty()) {
      command_ = setpoints_outer_.front().input;
    } else {
      command_ = Command(t);
    }

    // Managing expired references.
    if (sampler_->isTimeBased() && references_.front()->getEndTime() < t) {
      if (references_.size() > 1) {
        ReferenceVector::iterator first = references_.begin();
        logger_.info("Removing old reference: %s", (*first)->name().c_str());
        references_.erase(first);
      } else {
        QuadState hover =
          references_.front()->getEndSetpoint().state.getHoverState();
        if (!std::isfinite(hover.t))
          hover.t = references_.front()->getStartTime();
        hover.t = std::max(hover.t, t);
        appendReference(std::make_shared<HoverReference>(hover));
      }
    }
  }

  // Check if feedthrough is available
  bool feedthrough_valid = feedthrough_command_.valid();
  if ((t - feedthrough_command_.t) > feedthrough_timeout_) {
    logger_.warn("Feedthrough command timed out: %.3f!",
                 (t - feedthrough_command_.t));
    feedthrough_valid = false;
  }

  // If available, apply feedthrough command...
  apply_command_ = feedthrough_valid ? feedthrough_command_ : command_;

  // ... and set references to state estimate,
  // so that a time will break into hover.
  if (feedthrough_valid) {
    if (stop_after_feedthrough_) {
      references_.clear();
      references_.emplace_back(std::make_shared<HoverReference>(state_));
    }
  } else {
    clearFeedthoughCommand();
  }

  // Send command
  if (!bridge_ || !bridge_->send(apply_command_)) {
    logger_.error("Bridge failed!");
    return false;
  } else {
    bridge_->getFeedback(&feedback_);
  }

  // Spawn callbacks
  for (const PipelineCallbackFunction& callback : callbacks_)
    callback(state_, feedback_, references_, setpoints_, setpoints_outer_,
             setpoints_inner_, command_);

  return true;
}

bool Pipeline::appendReference(std::shared_ptr<ReferenceBase>&& reference) {
  if (!reference->valid()) {
    logger_.error("Reference is not valid!");
    return false;
  }

  if (references_.empty()) {
    references_.push_back(reference);
  } else {
    const Scalar start_time = reference->getStartTime();
    const Scalar trajectories_end_time = references_.back()->getEndTime();
    if (start_time >= trajectories_end_time) {
      references_.push_back(reference);
    } else if (references_.back()->isHover() &&
               references_.back()->isTimeInRange(start_time)) {
      references_.back()->truncate(start_time);
      references_.push_back(reference);
    } else {
      logger_.warn(
        "Not adding reference, since start time before current end time: "
        "t_start "
        "= %.2f, t_end = %.2f",
        start_time, trajectories_end_time);
      if (!references_.back()->isHover())
        logger_.warn("Last reference is not hover!");
      if (!references_.back()->isTimeInRange(start_time))
        logger_.warn("Time not in range!");
      return false;
    }
  }

  printReferenceInfo(true);
  return true;
}

bool Pipeline::insertReference(std::shared_ptr<ReferenceBase>&& reference) {
  // open for discussion: do we want a separate function for this? we could
  // disable this interface for the standard user maybe?
  if (!reference->valid()) {
    logger_.error("Reference is not valid!");
    return false;
  }

  if (references_.empty()) {
    references_.push_back(reference);
  } else {
    const Scalar start_time = reference->getStartTime();
    const Scalar trajectories_end_time = references_.back()->getEndTime();
    if (start_time >= trajectories_end_time) {
      references_.push_back(reference);
    } else {
      logger_.warn("Inserting reference at %.2f", start_time);

      ReferenceVector::const_iterator first_obsolete_ref = std::lower_bound(
        references_.begin(), references_.end(), start_time,
        [](const std::shared_ptr<ReferenceBase>& ref, const Scalar t) -> bool {
          return ref.get()->getStartTime() < t;
        });

      ReferenceVector new_references(references_.cbegin(), first_obsolete_ref);
      if (!new_references.empty()) references_.back()->truncate(start_time);

      new_references.push_back(reference);
      references_ = new_references;
    }
  }

  printReferenceInfo(true);
  return true;
}

void Pipeline::printReferenceInfo(const bool detailed) const {
  logger_.info("%zu references set%s:", references_.size(),
               detailed ? "" : ", first is");
  for (const std::shared_ptr<ReferenceBase>& reference : references_) {
    logger_ << *reference;
    if (!detailed) break;
  }
}

void Pipeline::registerCallback(const PipelineCallbackFunction& function) {
  callbacks_.push_back(function);
}

void Pipeline::setOuterloopDivisor(const int divisor) {
  outerloop_divisor_ = divisor;
}

void Pipeline::setStopAfterFeedthrough(const bool stop_after_feedthrough) {
  stop_after_feedthrough_ = stop_after_feedthrough;
}

void Pipeline::setFeedthroughTimeout(const Scalar& feedthrough_timeout) {
  feedthrough_timeout_ = feedthrough_timeout;
}

}  // namespace agi
