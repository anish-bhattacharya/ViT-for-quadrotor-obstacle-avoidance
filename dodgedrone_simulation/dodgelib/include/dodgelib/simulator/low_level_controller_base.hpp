#pragma once

#include "dodgelib/base/module.hpp"
#include "dodgelib/base/parameter_base.hpp"
#include "dodgelib/math/math.hpp"
#include "dodgelib/math/types.hpp"
#include "dodgelib/types/command.hpp"
#include "dodgelib/types/quad_state.hpp"
#include "dodgelib/types/quadrotor.hpp"

namespace agi {

class LowLevelControllerBase : public Module<LowLevelControllerBase> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 public:
  LowLevelControllerBase(const Quadrotor& quad,
                         const std::string& name = "LowLevelControllerBase");
  virtual ~LowLevelControllerBase() = default;
  virtual bool updateQuad(const Quadrotor&);
  virtual bool setState(const QuadState& state);
  virtual bool setCommand(const Command& cmd);
  virtual bool getMotorCommand(Ref<Vector<4>> motors);
  virtual bool setParamDir(const fs::path& param_dir);


 protected:
  // Command
  Command cmd_;

  // State of Quadrotor
  QuadState state_;

  // Motor speeds calculated by the controller
  Vector<4> motor_omega_des_;

  // Quadcopter to which the controller is applied
  Quadrotor quad_;

  // Directory to load parameters or thrust maps
  fs::path param_dir_;

  // Method that runs controller
  virtual void run() = 0;
};


}  // namespace agi
