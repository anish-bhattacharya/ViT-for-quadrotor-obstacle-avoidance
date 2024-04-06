#pragma once

#include <string>

#include "dodgelib/math/types.hpp"
#include "dodgelib/types/quad_state.hpp"
#include "dodgelib/types/quadrotor.hpp"
#include "dodgelib/utils/yaml.hpp"


namespace agi {

class ModelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ModelBase(Quadrotor& quad);
  virtual ~ModelBase() = default;

  virtual bool setParameters(const fs::path& filename);
  virtual bool setParameters(const Yaml&);
  virtual bool updateQuad(const Quadrotor&);

  // IN: State x
  // IN-OUT: Derivative of State
  virtual bool run(const Ref<const Vector<QS::SIZE>>,
                   Ref<Vector<QS::SIZE>>) const = 0;

 protected:
  Quadrotor quad_;
};

}  // namespace agi
