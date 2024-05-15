#include "dodgelib/simulator/model_base.hpp"

namespace agi {

ModelBase::ModelBase(Quadrotor& quad) : quad_(quad) {}

bool ModelBase::updateQuad(const Quadrotor& quad) {
  if (!quad.valid()) return false;
  quad_ = quad;
  return true;
}

bool ModelBase::setParameters(const fs::path& filename) {
  Yaml node{filename};
  return setParameters(node);
}

bool ModelBase::setParameters(const Yaml& cfg) { return false; }

}  // namespace agi
