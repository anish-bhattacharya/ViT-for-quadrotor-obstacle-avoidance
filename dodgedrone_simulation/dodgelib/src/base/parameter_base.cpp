#include "dodgelib/base/parameter_base.hpp"

#include <fstream>

namespace agi {

bool ParameterBase::load(const fs::path& filename) {
  return load(Yaml(filename));
}

bool ParameterBase::load(const Yaml& node) {
  throw ParameterException("ParameterBase load should not be called!");
  return false;
}

bool ParameterBase::valid() const { return false; }

}  // namespace agi
