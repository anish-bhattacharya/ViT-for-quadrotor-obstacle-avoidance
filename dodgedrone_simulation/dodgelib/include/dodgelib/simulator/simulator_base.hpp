#pragma once

// agi
#include "dodgelib/base/module.hpp"
#include "dodgelib/types/quad_state.hpp"

namespace agi {

class SimulatorBase : public Module<SimulatorBase> {
 public:
  SimulatorBase(const std::string& name = "Simulator");
  virtual ~SimulatorBase();

  // reset
  virtual bool reset(const bool& reset_time = true) = 0;
  virtual bool reset(const QuadState& state) = 0;

  // run the simulator
  virtual bool run(const Scalar dt) = 0;

  // public get function
  virtual bool getState(QuadState* const state) const = 0;

  // public set function
  virtual bool setState(const QuadState& state) = 0;
};

}  // namespace agi
