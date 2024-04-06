#include "dodgelib/sampler/time_based/time_sampler.hpp"

namespace agi {
TimeSampler::TimeSampler(const int horizon_len, const Scalar horizon_dt)
  : SamplerBase("Time Sampler", horizon_len, horizon_dt) {}

TimeSampler::~TimeSampler() {}

bool TimeSampler::getAt(const QuadState &state,
                        const ReferenceVector &references,
                        SetpointVector *const setpoints,
                        int horizon_len) const {
  // some sanity checks of inputs
  if (!state.valid() || setpoints == nullptr) return false;

  setpoints->clear();

  if (horizon_len < 0) {
    horizon_len = horizon_len_;
  }
  setpoints->reserve((size_t)horizon_len);


  ReferenceVector::const_iterator active_reference = std::lower_bound(
    references.begin(), references.end(), state.t,
    [](const std::shared_ptr<ReferenceBase> &ref, const Scalar t) -> bool {
      return ref.get()->getEndTime() <= t;
    });

  Scalar t_curr = state.t;
  if (active_reference == references.end()) {
    Setpoint setpoint = references.back()->getSetpoint(state);
    for (int i = 0; i < horizon_len; ++i) {
      setpoint.state.t += t_curr;
      setpoint.input.t += t_curr;
      setpoints->push_back(setpoint);
      t_curr += horizon_dt_;
    }
    return true;
  }

  Scalar t_ref_end = (*active_reference)->getEndTime();
  for (int i = 0; i < horizon_len; i++) {
    Setpoint setpoint;
    if (t_curr >= t_ref_end) {
      if (++active_reference >= references.end())
        active_reference = std::prev(active_reference);
      t_ref_end = (*active_reference)->getEndTime();
    }

    setpoints->emplace_back((*active_reference)->getSetpoint(state, t_curr));
    t_curr += horizon_dt_;
  }

  return true;
}


}  // namespace agi
