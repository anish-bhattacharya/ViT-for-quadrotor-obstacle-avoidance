#pragma once

#include "dodgelib/math/types.hpp"
#include "dodgelib/types/imu_sample.hpp"

namespace agi {

struct Feedback {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Scalar t{NAN};
  bool received{false};
  ImuSample imu;
  Quaternion attitude{NAN, NAN, NAN, NAN};

  Scalar voltage{NAN};
  Scalar current{NAN};

  Vector<4> rotor_speed_rads{NAN, NAN, NAN, NAN};
  Vector<4> rotor_thrust_newton{NAN, NAN, NAN, NAN};
  Vector<4> rotor_value{NAN, NAN, NAN, NAN};

  enum class CTRLMODE : uint8_t {
    INVALID = 0,
    OFF = 1,
    SET_PARAM = 2,
    ARM = 3,
    ROTOR_THROTTLE = 4,
    ROTOR_SPEED = 5,
    ROTOR_THRUST = 6,
    BODY_RATE = 7,
    ATTITUDE = 8,
    VELOCITY = 9,
    EMERGENCY = 10,
    FEEDBACK = 255
  };

  CTRLMODE control_mode{CTRLMODE::INVALID};
  bool armed{false};

  enum class ROTORFEEDBACKTYPE : uint8_t {
    INVALID = 0,
    THROTTLE = 1,
    SPEED = 2,
    THRUST = 3
  };

  ROTORFEEDBACKTYPE rotor_feedback_type{ROTORFEEDBACKTYPE::INVALID};

  inline bool valid() const { return std::isfinite(t) || received; }

  inline bool hasTimestamp() const { return std::isfinite(t); }

  inline bool isBatteryVoltageValid() const { return std::isfinite(voltage); }

  inline bool isBatteryCurrentValid() const { return std::isfinite(current); }

  inline bool isRotorSpeedValid() const { return rotor_speed_rads.allFinite(); }

  inline bool isImuValid() const { return imu.valid(); }

  inline bool isAttitudeValid() const { return attitude.coeffs().allFinite(); }
};

}  // namespace agi
