#pragma once

#include <memory>

#include "dodgelib/base/module.hpp"
#include "dodgelib/reference/reference_base.hpp"
#include "dodgelib/types/quad_state.hpp"
#include "dodgelib/types/setpoint.hpp"

namespace agi {

class SamplerBase : public Module<SamplerBase> {
 public:
  SamplerBase(const std::string& name = "Sampler", const int horizon_len = 10,
              const Scalar horizon_dt = 0.1);
  virtual ~SamplerBase();

  virtual bool getAt(const QuadState& state, const ReferenceVector& references,
                     SetpointVector* const setpoint,
                     int horizon_len = -1) const = 0;

  int getHorizonLength() const;

  Scalar getHorizonDt() const;

  bool getFull(const std::shared_ptr<ReferenceBase>& reference,
               SetpointVector* const setpoints) const;

  virtual bool isTimeBased() const = 0;

 protected:
  int horizon_len_;
  Scalar horizon_dt_;
};

}  // namespace agi