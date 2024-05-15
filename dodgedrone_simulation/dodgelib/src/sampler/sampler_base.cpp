#include "dodgelib/sampler/sampler_base.hpp"

namespace agi {

SamplerBase::SamplerBase(const std::string& name, const int horizon_len,
                         const Scalar horizon_dt)
  : Module(name), horizon_len_(horizon_len), horizon_dt_(horizon_dt) {}

SamplerBase::~SamplerBase() {}

int SamplerBase::getHorizonLength() const { return horizon_len_; }

Scalar SamplerBase::getHorizonDt() const { return horizon_dt_; }

bool SamplerBase::getFull(const std::shared_ptr<ReferenceBase>& trajectory,
                          SetpointVector* const setpoints) const {
  // some sanity checks of inputs
  if (setpoints == nullptr) {
    return false;
  }
  setpoints->clear();

  QuadState query_state = trajectory->getStartSetpoint().state;
  if (trajectory->isHover() || std::isinf(trajectory->getEndTime())) {
    setpoints->emplace_back(trajectory->getSetpoint(query_state));
  } else {
    for (; query_state.t <= trajectory->getEndTime();
         query_state.t += horizon_dt_) {
      setpoints->emplace_back(trajectory->getSetpoint(query_state));
    }
  }

  return true;
}

}  // namespace agi
