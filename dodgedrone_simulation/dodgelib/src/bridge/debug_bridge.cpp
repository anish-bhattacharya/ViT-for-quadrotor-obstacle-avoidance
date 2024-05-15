#include "dodgelib/bridge/debug_bridge.hpp"

#include "dodgelib/math/gravity.hpp"
#include "dodgelib/utils/throttler.hpp"

namespace agi {

DebugBridge::DebugBridge(const std::string& name,
                         const TimeFunction time_function)
  : BridgeBase(name, time_function) {}

bool DebugBridge::sendCommand(const Command& command, const bool active) {
  static Throttler timer_info_throttler(logger_, 1.0);
  if (!active) return true;

  timer_.toc();
  if (timer_() > 1.0) {
    logger_.info("Reset control dt timer.");
    timer_.reset();
    timer_.tic();
  }
  if (timer_.count() > 1)
    timer_info_throttler(&Logger::info, "Control dt: %1.3fs", timer_());

  return true;
}

void DebugBridge::reset() {
  timer_.reset();
  n_timeouts_ = 0;
}

}  // namespace agi
