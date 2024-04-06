#include "dodgelib/reference/reference_base.hpp"

#include <iomanip>
#include <iostream>

#include "dodgelib/math/gravity.hpp"

namespace agi {

ReferenceBase::ReferenceBase(const std::string& name) : name_(name) {}

ReferenceBase::ReferenceBase(const QuadState& state, const Scalar duration,
                             const std::string& name)
  : start_state_(state),
    duration_(duration),
    name_(name) {}  // TODO: , logger_(name) {}

ReferenceBase::~ReferenceBase() {}

Setpoint ReferenceBase::getSetpoint(const Scalar t, const Scalar heading) {
  QuadState query_state;
  query_state.t = t;
  if (std::isfinite(heading)) query_state.q(heading);
  return getSetpoint(query_state, t);
}

Setpoint ReferenceBase::getSetpoint(const QuadState& state) {
  return getSetpoint(state, state.t);
}

Setpoint ReferenceBase::getStartSetpoint() {
  Setpoint setpoint;
  setpoint.state = start_state_;
  setpoint.input = Command(start_state_.t, G, Vector<3>::Zero());
  return setpoint;
}

Setpoint ReferenceBase::getEndSetpoint() {
  Setpoint setpoint;
  setpoint.state = start_state_;
  setpoint.state.t = start_state_.t + duration_;
  setpoint.input = Command(start_state_.t + duration_, G, Vector<3>::Zero());
  return setpoint;
}

bool ReferenceBase::truncate(const Scalar& t) {
  if (t <= start_state_.t) return false;
  duration_ = t - start_state_.t;
  return true;
}

bool ReferenceBase::isTimeInRange(const Scalar time) const {
  return time >= getStartTime() && time <= getEndTime();
}

std::string ReferenceBase::time_in_HH_MM_SS_MMM(const Scalar time) const {
  if (!std::isfinite(time)) return "inf";

  const unsigned int ms = fmod(time, 1000.0);

  // convert to std::time_t in order to convert to std::tm (broken time)
  std::time_t timer = (std::time_t)(time);

  // convert to broken time
  std::tm bt = *std::localtime(&timer);

  std::ostringstream oss;

  oss << std::put_time(&bt, "%H:%M:%S");  // HH:MM:SS
  oss << '.' << std::setfill('0') << std::setw(3) << ms;

  return oss.str();
}

std::ostream& operator<<(std::ostream& os, const ReferenceBase& ref) {
  os << std::setw(30) << std::left << ref.name() << "  | "
     << ref.time_in_HH_MM_SS_MMM(ref.getStartTime()) << " --> "
     << ref.getDuration() << "s --> "
     << ref.time_in_HH_MM_SS_MMM(ref.getEndTime()) << " |" << std::endl;
  return os;
}
}  // namespace agi
