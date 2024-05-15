#pragma once

#include "dodgelib/bridge/bridge_base.hpp"
#include "dodgelib/utils/timer.hpp"

namespace agi {

class DebugBridge : public BridgeBase {
 public:
  DebugBridge(const std::string& name, const TimeFunction time_function);
  void reset() override;

 protected:
  void guardTimeout() override {}
  virtual bool sendCommand(const Command& command, const bool active) override;

  Timer timer_{"Sending dt"};
};

}  // namespace agi
