#include "dodgelib/controller/controller_base.hpp"

namespace agi {

ControllerBase::ControllerBase(const std::string& name, const Scalar exec_dt,
                               const int horizon_length)
  : Module(name),
    horizon_length_(horizon_length),
    exec_dt_(exec_dt),
    pred_dt_(NAN) {}

ControllerBase::~ControllerBase() {}

}  // namespace agi
