#pragma once

#include <memory>
#include <string>
#include <vector>

#include "dodgelib/base/module.hpp"
#include "dodgelib/math/types.hpp"
#include "dodgelib/types/setpoint.hpp"
#include "dodgelib/utils/logger.hpp"

namespace agi {

class ReferenceBase {
 public:
  ReferenceBase(const std::string& name = "ReferenceBase");
  ReferenceBase(const QuadState& state, const Scalar duration,
                const std::string& name = "ReferenceBase");
  virtual ~ReferenceBase();

  virtual Setpoint getSetpoint(const QuadState& state, const Scalar t) = 0;
  virtual Setpoint getSetpoint(const QuadState& state) final;
  virtual Setpoint getSetpoint(const Scalar t,
                               const Scalar heading = NAN) final;
  virtual Setpoint getStartSetpoint();
  virtual Setpoint getEndSetpoint();

  inline Scalar getStartTime() const { return start_state_.t; }
  inline Scalar getEndTime() const { return start_state_.t + duration_; }
  inline Scalar getDuration() const { return duration_; }
  bool isTimeInRange(const Scalar time) const;
  std::string time_in_HH_MM_SS_MMM(const Scalar time) const;

  virtual bool valid() const { return start_state_.valid(); }
  const std::string& name() const { return name_; }
  virtual bool isHover() const { return false; }
  virtual bool isVelocityRefernce() const { return false; }
  virtual bool isAbsolute() const { return true; }

  bool truncate(const Scalar& t);
  friend std::ostream& operator<<(std::ostream& os, const ReferenceBase& ref);

 protected:
  QuadState start_state_;
  Scalar duration_;
  const std::string name_;
};

using ReferenceVector = std::vector<std::shared_ptr<ReferenceBase>>;

}  // namespace agi
