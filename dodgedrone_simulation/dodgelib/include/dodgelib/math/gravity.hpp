#pragma once
#include "dodgelib/math/types.hpp"

namespace agi {

/**
 * Gravity Value [m/s^2]
 *
 * This is the gravity value used in the whole project.
 */
static constexpr Scalar G = 9.8066;


/**
 * Gravity Vector [m/s^2]
 *
 * This is the gravity vector pointing in negative z-direction.
 * It uses the value of #G.
 */
const Vector<3> GVEC{0, 0, -G};

}  // namespace agi
