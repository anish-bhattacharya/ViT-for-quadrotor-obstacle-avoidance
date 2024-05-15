#pragma once

#include <memory>

#include "dodgelib/types/command.hpp"
#include "dodgelib/types/quad_state.hpp"

namespace agi {

struct Setpoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Setpoint() = default;
  Setpoint(const QuadState& state, const Command& input)
    : state(state), input(input) {}

  QuadState state;
  Command input;
};

using SetpointVector = std::vector<Setpoint>;

}  // namespace agi
