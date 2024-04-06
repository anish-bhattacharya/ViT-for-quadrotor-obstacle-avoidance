#pragma once

#include <memory>

#include "dodgelib/base/module.hpp"
#include "dodgelib/types/command.hpp"
#include "dodgelib/types/imu_sample.hpp"
#include "dodgelib/types/quad_state.hpp"
#include "dodgelib/types/setpoint.hpp"

namespace agi {

class ControllerBase : public Module<ControllerBase> {
 public:
  ControllerBase(const std::string& name = "Controller",
                 const Scalar exec_dt = 0.01, const int horizon_length = 1);
  virtual ~ControllerBase();

  virtual bool getCommand(const QuadState& state,
                          const SetpointVector& reference,
                          SetpointVector* const setpoints) = 0;

  virtual void addImuSample(const ImuSample& sample) {}

  Scalar dt() const { return pred_dt_; }
  int horizonLength() const { return horizon_length_; }

 protected:
  int horizon_length_;
  Scalar exec_dt_;  // dt at which we will run the controller
  Scalar pred_dt_;  // dt of the predictions
};

}  // namespace agi
