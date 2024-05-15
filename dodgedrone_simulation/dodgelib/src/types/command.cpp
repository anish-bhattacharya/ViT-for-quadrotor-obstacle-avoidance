#include "dodgelib/types/command.hpp"

#include <cmath>

namespace agi {

Command::Command() {}

Command::Command(const Scalar t, const Scalar thrust, const Vector<3>& omega)
  : t(t), collective_thrust(thrust), omega(omega) {}

Command::Command(const Scalar t, const Vector<4>& thrusts)
  : t(t), thrusts(thrusts) {}

Command::Command(const Scalar t)
  : t(t),
    collective_thrust(0.0),
    omega(Vector<3>::Zero()),
    thrusts(Vector<4>::Zero()) {}

bool Command::valid() const {
  return std::isfinite(t) &&
         ((std::isfinite(collective_thrust) && omega.allFinite()) ||
          thrusts.allFinite());
}

bool Command::isSingleRotorThrusts() const {
  return std::isfinite(t) && thrusts.allFinite();
}

bool Command::isRatesThrust() const {
  return std::isfinite(t) && std::isfinite(collective_thrust) &&
         omega.allFinite();
}

std::ostream& operator<<(std::ostream& os, const Command& command) {
  os.precision(3);
  os << std::scientific;
  os << "Command at " << command.t << "s:\n"
     << "collective_thrust =  [" << command.collective_thrust << "]\n"
     << "omega =  [" << command.omega.transpose() << "]\n"
     << "thrusts =  [" << command.thrusts.transpose() << "]" << std::endl;
  os.precision();
  os.unsetf(std::ios::scientific);
  return os;
}

bool Command::operator==(const Command& rhs) const {
  if (t != rhs.t) {
    return false;
  }
  if (isRatesThrust()) {
    return (collective_thrust == rhs.collective_thrust &&
            omega.isApprox(rhs.omega, 1e-3));
  } else {
    return (thrusts.isApprox(rhs.thrusts, 1e-3));
  }
}


}  // namespace agi
