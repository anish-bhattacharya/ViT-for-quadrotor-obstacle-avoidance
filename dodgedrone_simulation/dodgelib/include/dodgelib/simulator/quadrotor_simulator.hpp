#pragma once

// std
#include <memory>
#include <type_traits>

// dodgelib
#include "dodgelib/base/module.hpp"
#include "dodgelib/math/integrator_rk4.hpp"
#include "dodgelib/math/types.hpp"
#include "dodgelib/simulator/low_level_controller_simple.hpp"
#include "dodgelib/simulator/model_base.hpp"
#include "dodgelib/simulator/model_init.hpp"
#include "dodgelib/simulator/model_motor.hpp"
#include "dodgelib/simulator/model_rigid_body.hpp"
#include "dodgelib/simulator/model_thrust_torque_simple.hpp"
#include "dodgelib/simulator/simulator_base.hpp"
#include "dodgelib/types/command.hpp"
#include "dodgelib/types/quad_state.hpp"
#include "dodgelib/types/quadrotor.hpp"

namespace agi {

class QuadrotorSimulator : public SimulatorBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  QuadrotorSimulator(const Quadrotor &quad = Quadrotor(1.0, 0.25));

  // reset
  bool reset(const bool &reset_time = true) override;
  bool reset(const QuadState &state) override;

  // run the quadrotor simulator
  bool run(const Scalar ctl_dt) override;
  bool run(const Command &cmd, const Scalar ctl_dt);

  // public get functions
  bool getState(QuadState *const state) const override;
  bool getQuadrotor(Quadrotor *const quad) const;
  const Quadrotor &getQuadrotor() const;
  const std::shared_ptr<LowLevelControllerBase> getLowLevelController() const;

  // public set functions
  bool setState(const QuadState &state) override;
  bool setCommand(const Command &cmd);
  bool updateQuad(const Quadrotor &quad);
  bool setLowLevelController(const std::string &llc_name);

  // public pipeline construction
  template<class T>
  std::shared_ptr<T> addModel(const T &&mdl) {
    static_assert(std::is_base_of<ModelBase, T>::value,
                  "Model must be derived from ModelBase");
    model_pipeline_.push_back(std::make_shared<T>(mdl));
    return std::dynamic_pointer_cast<T>(model_pipeline_.back());
  }


 private:
  // quadrotor dynamics, integrator
  Quadrotor quadrotor_;
  std::shared_ptr<LowLevelControllerBase> ctrl_;
  std::vector<std::shared_ptr<ModelBase>> model_pipeline_;
  std::unique_ptr<IntegratorRK4> integrator_ptr_;

  // control command
  Command cmd_;

  // quadrotor state
  QuadState state_;

  void updateState(const QuadState &, Scalar);
  DynamicsFunction getDynamics() const;
  bool computeDynamics(const Ref<const Vector<QS::SIZE>>,
                       Ref<Vector<QS::SIZE>>) const;
};


}  // namespace agi
