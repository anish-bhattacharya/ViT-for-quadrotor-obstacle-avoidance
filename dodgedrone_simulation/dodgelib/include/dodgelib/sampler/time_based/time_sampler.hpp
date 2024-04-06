#pragma once

#include "dodgelib/sampler/sampler_base.hpp"

namespace agi {

class TimeSampler : public SamplerBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  TimeSampler(const int horizon_len, const Scalar horizon_dt);
  TimeSampler() : TimeSampler(10, 0.1){};
  ~TimeSampler();

  bool getAt(const QuadState &state, const ReferenceVector &references,
             SetpointVector *const setpoints, int horizon_len = -1) const;

  bool isTimeBased() const { return true; }
};

}  // namespace agi