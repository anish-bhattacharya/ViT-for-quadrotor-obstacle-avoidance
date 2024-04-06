#include "dodgelib/bridge/bridge_base.hpp"

namespace agi {
BridgeBase::BridgeBase(const std::string& name,
                       const TimeFunction time_function, const Scalar timeout,
                       const int n_max_timeouts)
  : Module(name),
    timeout_(timeout),
    n_max_timeouts_(n_max_timeouts),
    time_function_(time_function) {
  if (timeout_ > 0.0)
    timeout_guard_thread_ = std::thread(&BridgeBase::guardTimeout, this);
  else
    logger_.warn("Not starting guard!");
}

BridgeBase::~BridgeBase() {
  shutdown_ = true;
  if (timeout_guard_thread_.joinable()) timeout_guard_thread_.join();
}

bool BridgeBase::send(const Command& command) {
  // First check if timeout has locked out.
  if (locked()) return false;

  got_command_ = true;
  const bool ret = sendCommand(command, active_);

  // If command is successfully sent, reset timeout.
  // Decrease timeout counter instead of resetting it to catch
  // critically low send rates.
  if (ret) {
    timeout_reset_cv_.notify_all();
    if (n_timeouts_ > 0) --n_timeouts_;
  }

  return ret;
}

void BridgeBase::guardTimeout() {
  const std::chrono::milliseconds timeout((int)(timeout_ * 1000));
  std::unique_lock<std::mutex> lock(timeout_wait_mutex_);

  while (!shutdown_) {
    // Wait for timeout. If notified during waiting, nothing happens.
    if (timeout_reset_cv_.wait_for(lock, timeout) == std::cv_status::timeout) {
      if (n_max_timeouts_ < 1 || !active_ || !got_command_) {
        const Command zero_command(time_function_());
        sendCommand(zero_command, active_);
        continue;  // not armed, or no command yet
      }

      // If we have not yet reached critical number of timeouts,
      // we can leave the platform armed and hope the user catches it.
      if (!locked()) {
        ++n_timeouts_;
        const Command zero_command(time_function_());
        sendCommand(zero_command, active_);
      } else {  // Else we disarm.
        deactivate();
        const Command zero_command(time_function_());
        sendCommand(zero_command, active_);
      }
    }
  }
}

bool BridgeBase::activate() {
  if (locked()) {
    logger_.warn("Can't activate because locked!");
    return false;
  }

  active_ = true;
  logger_.info("Activated!");

  return active_;
}

bool BridgeBase::deactivate() {
  active_ = false;
  logger_.info("Deactivated!");
  return active_;
}

void BridgeBase::setVoltage(const Scalar voltage) {
  latest_raw_voltage = voltage;
  voltage_ += voltage;
}

Scalar BridgeBase::getVoltage() const { return voltage_.get(); }

void BridgeBase::reset() {
  this->n_timeouts_ = 0;
  got_command_ = false;
}

bool BridgeBase::active() const { return active_; }

bool BridgeBase::locked() const {
  return n_max_timeouts_ > 0 && n_timeouts_ >= n_max_timeouts_;
}

bool BridgeBase::getFeedback(Feedback* const feedback) {
  if (feedback == nullptr) return false;
  feedback->t = time_function_();
  feedback->armed = active_;
  return true;
}

void BridgeBase::registerFeedbackCallback(FeedbackCallbackFunction function) {
  feedback_callbacks_.push_back(function);
}

}  // namespace agi
